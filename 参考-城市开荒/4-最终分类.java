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

// ---- 4. Final LCC ---- //
// The final LCC script is almost exactly the same as the 3--InitialLCC, the only adjustments are in the numbers of points generated from non-CK and CK sites.

/* REQUIREMENTS

var ctr: pick a point around the center of your study area
var city: import the boundary of your study area 
var final_bands: import the prepared dataset from 2--DatasetPreparation into the script
var NK: import shapefile of non-CK sites to generate training and verification points. For the paper non-CK sites were chosen specifically to train the classifier 
        to differentiate between CK and other similar urban land uses, such as commercial agriculture, parks, wetlands (see Extended Data Table 2 for more details)
var CK: import shapefile of the now larger sample of CK sites to generate training and verification points.

PLEASE NOTE: Vector datasets of CK locations collected/detected in this study are not included in this script to protect the right to privacy of individual practitioners 
and prevent their CK sites from being targeted. This is done in accordance with the study's ethics approval requirements. If you require access to the data, please
contact the author.

*/

// Import training dataset
var img = final_bands;
var new_bands = final_bands.bandNames().remove("clusters");

//Import regions to generate training and verification points

var not_kaihuang = ee.FeatureCollection.randomPoints({
    region: NK.filterBounds(city),
    points: 11000, //For each year, the amount of points generated is determined after multiple iterations of different values - the one chosen yielded the highest accuracies
    seed: 0
  });

// Label non-CK points as landcover 1
var NKLabel = function(feature) {
  return feature.set({landcover: 1});
};

// Randomly distribute points for training (70%) and for verification (30%) 
var not_kaihuang = not_kaihuang.map(NKLabel).randomColumn();

var not_kaihuang70 = not_kaihuang.filter('random >= 0.3');
var not_kaihuang30 = not_kaihuang.filter('random < 0.3');

var KLabel = function(feature) {
  return feature.set({landcover: 0});
};

// Generate CK points
var UH_pts = ee.FeatureCollection.randomPoints({
    region: CK,
    points: 636,
    seed: 0
  });

var kaihuang = UH_pts.map(KLabel).randomColumn();

var kaihuang70 = kaihuang.filter('random >= 0.3');
Map.addLayer(kaihuang70)
var kaihuang30 = kaihuang.filter('random < 0.3');

// Compile training points dataset
var t_points = ee.FeatureCollection(not_kaihuang70).merge(kaihuang70);
// Map.addLayer(t_points);

//sample the input imagery to get a FeatureCollection of training data
//the reflectance value of each band is now stored along with class label for every training point
var training = img.sampleRegions({
            collection: t_points, // your training data
            properties: ['landcover'], // the property you defined
            scale : 10 // spatial resolution according to S2 data (if it is really slow you can set it to 20)
});

// Make a Random Forest classifier and train it.
var classifier =  ee.Classifier.smileRandomForest(10).train({
  features : training,                    // what to use as my training data
  classProperty: 'landcover',            //what categories - classes labelled in training data
  inputProperties: new_bands                  //what bands
});

// Classify the rest of the image
var classified = img.classify(classifier);

// Define a palette for the Land Use classification.
// search for palettes on the internet

var my_palette = [
  '#FF0000', /// CK // red 
  '#0000FF' /// non-CK // blue       
];

// Display the classification result and the input image. 
//MAX is the number of classes to consider
Map.addLayer(classified, {min:0,max:1, palette:my_palette}, 'Landcover');

Export.image.toDrive({
  image: classified,
  region: city,
  fileNamePrefix: 'classified',
  folder: 'GEE',
  crs: 'EPSG:5070',
  scale:10,
  maxPixels:1e10
});

// ---- Accuracy Assessment 
// Code adapted from GEARS lab tutorial, https://www.gears-lab.com/intro_rs_lab7/#classification-validation

// remaining 30% of points in a variable v_points
var v_points = ee.FeatureCollection(not_kaihuang30).merge(kaihuang30);
  
// Sample classfication results to validation points.
var validation = classified.sampleRegions({
  collection: v_points,
  properties: ['landcover'],
  scale: 10,
});

//Compare the landcover of validation points against the classification result
var testAccuracy = validation.errorMatrix('landcover', 'classification');
//Print the error matrix to the console
print('Validation error matrix: ', testAccuracy);
//Print the overall accuracy to the console
print('Validation overall accuracy: ', testAccuracy.accuracy());
//Print the consumer accuracy to the console
print('Validation consumer accuracy: ', testAccuracy.consumersAccuracy());
//Print the kappa accuracy to the console
print('Validation kappa accuracy: ', testAccuracy.kappa());
//Print the producer accuracy to the console
print('Validation producer accuracy: ', testAccuracy.producersAccuracy());

var exportAccuracy = ee.Feature(null, {matrix: testAccuracy.array()});


//Calculate area of Homesteads.
var chengshi_kaihuang = classified.eq(0);
Map.addLayer(chengshi_kaihuang.selfMask(), {min:0, max:1, palette: ['white', 'red']}, 'Homesteads');

var areaKaihuang = chengshi_kaihuang.multiply(ee.Image.pixelArea())

Export.image.toDrive({
  image: chengshi_kaihuang,
  region: city,
  fileNamePrefix: 'CK',
  folder: 'GEE', 
  crs: 'EPSG:5070',
  scale:10,
  maxPixels:1e10
});

var area = areaKaihuang.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: city,
  scale: 30,
  maxPixels: 1e10
  })
  
// Calculate percentage of CK area within study area, the number used below is the calculated value for the total land area of Wuhan's central urban districts (excluding bodies of water)
var CKArea = ee.Number(area.get('classification')).round();
var cityArea = ee.Number(826611368.1578786).round(); 
print(CKArea)

var CKPercentage = CKArea.divide(cityArea).multiply(100);
print(CKPercentage);
