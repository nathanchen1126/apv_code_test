/*
This code was developed for the paper 
"The Role of Informal Ruralization within China's Rapid Urbanization" by Hanxi Wang (hanxi.wang.22@ucl.ac.uk)
for the detection of chengshi kaihuang, a practice of informal urban wasteland cultivation, that has emerged in the central urban districts of Wuhan, China. 

The codes used for the study is divided into 4 sections as running the entire script in one go can exceed GEE's processing capacity.
The 4 sections are documented under this shared repository as:
1--BSI-SD (creates the band which provides information on how much demolition a certain site has experienced before a certain year)
2--DatasetPreparation (processes S2 data, adds spectral bands and object-based features to produce the final dataset used for classification)
3--InitialLCC (runs the first LCC based on a limited number of known sites obtained from grey literature)
4--FinalLCC (runs the final LCC based on a much larger sample of CK sites manually verified from the initial LCC results)
*/

// ---- 2. Dataset Preparation ---- //

/* REQUIREMENTS

var BSI_bands: import the BSI_SD band generated in 1--BSI_SD into the script

*/

//--- Define cloud masking function for Sentinel-2 data
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

//--- Filter and select S2 data for the study area in the target year (e.g. 2018)
var dataset = ee.ImageCollection('COPERNICUS/S2');

var wuhan = dataset.filterBounds(ctr)
            .filterDate('2018-04-01', '2018-09-30')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
            .map(maskS2clouds)
            .median()
            .clip(city);
            
print(wuhan);

// Select relevant bands
var S2Bands = ["B2", "B3", "B4","B5", "B6", "B8", "B11"];
var wuhan = wuhan.select(S2Bands);

var rgb = wuhan.select(['B4', 'B3', 'B2']);

// Export RGB image
Export.image.toDrive({
  image: rgb,
  region: city,
  fileNamePrefix: '2018_rgb',
  folder: 'GEE',
  crs: 'EPSG:5070',
  scale:10,
  maxPixels:1e10
});

var visualization = {
  min: 0,
  max: .25,
  bands: ['B4', 'B3', 'B2'],
};

Map.addLayer(wuhan, visualization, 'RGB');

// NDVI - define function and calculate band for object clustering 
// Generally a gray index is used for object clustering (see Tassi and Vizzari). NDVI is used here due to this paper's focus on vegetated areas)
var getNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

// --- Pixel-Based Features - Spectral Indices
var SIs = function(image) {
  var ndvi = getNDVI(image);
  var to = image.expression(
    '(3 * ((RE1 - R) - 0.2 * (RE1 - G) * (RE1 / R))) / (1.16 * (N - R) / (N + R + 0.16))', {
      'RE1': image.select('B5'),
      'R': image.select('B4'),
      'G': image.select('B3'),
      'N': image.select('B8')
  }).rename('TO');
  var evi = image.expression(
    '(2.5 * (N - R)) / (N + 6 * R - 7.5 * B + 1)', {
      'N': image.select('B8'),
      'R': image.select('B4'),
      'B': image.select('B2')
  }).rename('EVI');
  var bsi = image.expression(
    '((R + S1) - (N + B)) / ((R + S1) + (N + B))', {
    'N': image.select('B8'),
    'R': image.select('B4'),
    'S1': image.select('B11'),
    'B': image.select('B2')
  }).rename('BSI');
  var ndwi = image.expression(
    '(G - N) / (G + N)', {
      'N': image.select('B8'),
      'G': image.select('B3')
  }).rename('NDWI');
  var ndli = image.expression(
    '(G - R) / (G + R + S1)', {
      'G': image.select('B3'),
      'R': image.select('B4'),
      'S1': image.select('B11')
  }).rename('NDLI');
  return ndvi.addBands(ndwi)
              .addBands(evi)
              .addBands(bsi)
              .addBands(to)
              .addBands(ndli)
              .addBands(BSI_bands.select('BSI_sd'));
};

var wuhan = SIs(wuhan);
print(wuhan, 'Image with SIs');

//--- Object-Based Features - Textural Measures (GLCM) + Segmentation (SNIC)
// The section below is adapted from the code created by Tassi and Vizzari (2020)

// Select the NDVI band
var ndvi = getNDVI(wuhan).select('NDVI');
// Map.addLayer(ndvi, {min:-1, max: 1, palette: ['red', 'white', 'green']},"2023 NDVI");

//--- GLCM - calculate 18 bands of GLCM textural measures
var glcm = ndvi.multiply(100).toInt16().glcmTexture({size: 2});

// GLCM indices selected for PCA according to Tassi and Vizzari [to be changed if not satisfactory]
var glcm_bands= ["NDVI_asm","NDVI_contrast","NDVI_corr","NDVI_ent","NDVI_var","NDVI_idm","NDVI_savg"]

//--- Before the PCA the glcm bands are scaled
var image = glcm.select(glcm_bands);
// calculate the min and max value of an image
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: city,
  scale: 10,
  maxPixels: 10e9,
}); 
var glcm = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());

// Create function to compile PCA on GLCM measures into a single band
function PCA(maskedImage){
  var image = maskedImage.unmask()
  var scale = 20;
  var region = city;
  var bandNames = image.bandNames();
  // Mean center the data to enable a faster covariance reducer
  // and an SD stretch of the principal components.
  var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: city,
    scale: scale,
    maxPixels: 1e9,
    bestEffort: true,
    tileScale: 16
  });
  var means = ee.Image.constant(meanDict.values(bandNames));
  var centered = image.subtract(means);
  // This helper function returns a list of new band names.
  var getNewBandNames = function(prefix) {
    var seq = ee.List.sequence(1, bandNames.length());
    return seq.map(function(b) {
      return ee.String(prefix).cat(ee.Number(b).int());
    });
  };
  // This function accepts mean centered imagery, a scale and
  // a region in which to perform the analysis.  It returns the
  // Principal Components (PC) in the region as a new image.
  var getPrincipalComponents = function(centered, scale, region) {
    // Collapse the bands of the image into a 1D array per pixel.
    var arrays = centered.toArray();
    
    // Compute the covariance of the bands within the region.
    var covar = arrays.reduceRegion({
      reducer: ee.Reducer.centeredCovariance(),
      geometry: city,
      scale: scale,
      maxPixels: 1e9,
      bestEffort: true,
      tileScale: 16
    });
    // Get the 'array' covariance result and cast to an array.
    // This represents the band-to-band covariance within the region.
    var covarArray = ee.Array(covar.get('array'));
    // Perform an eigen analysis and slice apart the values and vectors.
    var eigens = covarArray.eigen();
    // This is a P-length vector of Eigenvalues.
    var eigenValues = eigens.slice(1, 0, 1);
    
    // Compute Percentage Variance of each component
    var eigenValuesList = eigenValues.toList().flatten()
    var total = eigenValuesList.reduce(ee.Reducer.sum())
    var percentageVariance = eigenValuesList.map(function(item) {
      return (ee.Number(item).divide(total)).multiply(100).format('%.2f')
    })
    // This will allow us to decide how many components capture
    // most of the variance in the input
    print('Percentage Variance of Each Component', percentageVariance)
    // This is a PxP matrix with eigenvectors in rows.
    var eigenVectors = eigens.slice(1, 1);
    // Convert the array image to 2D arrays for matrix computations.
    var arrayImage = arrays.toArray(1);
    // Left multiply the image array by the matrix of eigenvectors.
    var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
    // Turn the square roots of the Eigenvalues into a P-band image.
    var sdImage = ee.Image(eigenValues.sqrt())
      .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
    // Turn the PCs into a P-band image, normalized by SD.
    return principalComponents
      // Throw out an an unneeded dimension, [[]] -> [].
      .arrayProject([0])
      // Make the one band array image a multi-band image, [] -> image.
      .arrayFlatten([getNewBandNames('pc')])
      // Normalize the PCs by their SDs.
      .divide(sdImage);
  };
  var pcImage = getPrincipalComponents(centered, scale, region);
  return pcImage.mask(maskedImage.mask());
}

var pca = PCA(glcm).select(['pc1', 'pc2', 'pc3']);
// Map.addLayer(pca, {bands: ['pc1']}, 'PCA');

//--- SNIC
var size_segmentation = 15

var seeds = ee.Algorithms.Image.Segmentation.seedGrid(size_segmentation);

var snic = ee.Algorithms.Image.Segmentation.SNIC({
  image: wuhan, 
  compactness: 0,  
  connectivity: 8, 
  neighborhoodSize: 256, 
  seeds: seeds
})
// Map.addLayer(snic.randomVisualizer(), {}, 'SNIC Segment Clusters', true, 1);

var wuhan = wuhan.addBands(snic.select("clusters"));

//Select the band "clusters" from the snic output fixed on its scale of 10 meters and add them the PC1 taken from the PCA data.
// Calculate the mean for each segment with respect to the pixels in that cluster
var clusters_snic = snic.select("clusters")
clusters_snic = clusters_snic.reproject ({crs: clusters_snic.projection (), scale: 10});
//Map.addLayer(clusters_snic.randomVisualizer(), {}, 'clusters')

var new_feature = clusters_snic.addBands(pca.select("pc1"))

var new_feature_mean = new_feature.reduceConnectedComponents({
  reducer: ee.Reducer.mean(),
  labelBand: 'clusters'
})

//Create a dataset with the new band used so far together with the band "clusters" and their new mean parameters
var final_bands = new_feature_mean.addBands(snic) 
print(final_bands);
// Map.addLayer(final_bands, {bands: ['B2_mean', 'B3_mean', 'B4_mean'], max: 0.25}, 'final bands RGB')

Export.image.toAsset({
  image: final_bands,
  description: '2023_final_bands_L',
  assetId: '2023_final_bands_L',
  scale: 10,
  pyramidingPolicy: 'mode',
  region: city
});
