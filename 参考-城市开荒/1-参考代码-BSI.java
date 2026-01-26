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

// ---- 1. BSI_SD ---- //

/* REQUIREMENTS

var ctr: pick a point around the center of your study area
var city: import the boundary of your study area 

*/

// Load Landsat 5 ImageCollection
var collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2');
    
// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}

collection = collection.map(applyScaleFactors);

//Define RGB visualization settings
var visualization = {
  bands: ['SR_B3', 'SR_B2', 'SR_B1'],
  min: 0.0,
  max: 0.3,
};

// Create BSI function 
var getBSI = function(image) {
  return image.expression(
    '((R + S1) - (N + B)) / ((R + S1) + (N + B))', {
    'N': image.select('SR_B4'),
    'R': image.select('SR_B3'),
    'S1': image.select('SR_B5'),
    'B': image.select('SR_B1')
  });
};

// Create function to calculate BSI of a particular year's median cloudless mosaick image 
var L5_BSI = function(date) {
  var startdate = ee.String(date).cat('-01-01');
  var enddate = ee.String(date).cat('-12-30');
  var image = collection.filterDate(startdate, enddate)
                .filter(ee.Filter.lt('CLOUD_COVER',5))
                .median().clip(city);
  var img_bsi = getBSI(image);
  // Map.addLayer(image, visualization, 'True Color');
  // Map.addLayer(img_bsi, {min:-0.2, max:0.2, palette:['red','white', 'green']}, date);
  return img_bsi.rename('BSI');
};

//Generate BSI for each year between 2001 - 2011
var BSI_2001 = L5_BSI('2001');
var BSI_2002 = L5_BSI('2002');
var BSI_2003 = L5_BSI('2003');
var BSI_2004 = L5_BSI('2004'); 
var BSI_2005 = L5_BSI('2005');
var BSI_2006 = L5_BSI('2006');
var BSI_2007 = L5_BSI('2007');
var BSI_2008 = L5_BSI('2008');
var BSI_2009 = L5_BSI('2009');
var BSI_2010 = L5_BSI('2010'); 
var BSI_2011 = L5_BSI('2011'); 

//--- Landsat 8 2013-2023

var collectionL8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');

// Applies scaling factors.
function applyScaleFactorsL8(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

collectionL8 = collectionL8.map(applyScaleFactorsL8);

var visualizationL8 = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0.0,
  max: 0.3,
};

// BSI function 
var getBSIL8 = function(image) {
  return image.expression(
    '((R + S1) - (N + B)) / ((R + S1) + (N + B))', {
    'N': image.select('SR_B5'),
    'R': image.select('SR_B4'),
    'S1': image.select('SR_B6'),
    'B': image.select('SR_B2')
  });
};

// Yearly BSI calculation function
var L8_BSI = function(date) {
  var startdate = ee.String(date).cat('-01-01');
  var enddate = ee.String(date).cat('-12-30');
  var image = collectionL8.filterDate(startdate, enddate)
                .filter(ee.Filter.lt('CLOUD_COVER',5))
                .median().clip(city);
  var img_bsi = getBSIL8(image);
  // Map.addLayer(image, visualizationL8, 'True Color');
  // Map.addLayer(img_bsi, {min:-0.2, max:0.2, palette:['red','white', 'green']}, date);
  return img_bsi.rename('BSI');
};

//Generate BSI for each year between 20013 - 2022
var BSI_2013 = L8_BSI('2013');
var BSI_2014 = L8_BSI('2014');
var BSI_2015 = L8_BSI('2015');
var BSI_2016 = L8_BSI('2016');
var BSI_2017 = L8_BSI('2017');
var BSI_2018 = L8_BSI('2018');
var BSI_2019 = L8_BSI('2019');
var BSI_2020 = L8_BSI('2020'); 
var BSI_2021 = L8_BSI('2021'); 
var BSI_2022 = L8_BSI('2022'); 

// For each year where LCC will be performed (years where S2 data is available), compile available BSI images of previous years
var Combined2016 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016]);
var Combined2017 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017]);
var Combined2018 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017, BSI_2018]);
var Combined2019 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017, BSI_2018, BSI_2019]);
var Combined2020 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017, BSI_2018, BSI_2019, BSI_2020]);
var Combined2021 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017, BSI_2018, BSI_2019, BSI_2020, BSI_2021]);
var Combined2022 = ee.ImageCollection([BSI_2001, BSI_2002, BSI_2003, BSI_2004, BSI_2005, BSI_2006, BSI_2007, BSI_2008, BSI_2009, BSI_2010, BSI_2011, BSI_2013, BSI_2014, BSI_2015, BSI_2016, BSI_2017, BSI_2018, BSI_2019, BSI_2020, BSI_2021, BSI_2022]);

var vacantStatus = function(image) {
  return image.gt(0);
};

//Create function to calculate the standard deviation of previous BSI values for any particular year
var AdditionalBands = function (dataset,year) {
  var BSI_hist = dataset.map(vacantStatus).sum().rename('BSI_hist');
  var BSI_sd = dataset.reduce(ee.Reducer.stdDev()).rename('BSI_sd');
  var BSI_bands = BSI_hist.addBands(BSI_sd)
  Export.image.toAsset({
  image: BSI_bands,
  description: year,
  assetId: year,
  scale: 10,
  pyramidingPolicy: 'mode',
  region: city
});
  return BSI_bands;
};

// Calculate and export BSI_SD bands for 2016-2022
var Bands_2016 = AdditionalBands(Combined2016, 'BSI_bands_2016');
var Bands_2017 = AdditionalBands(Combined2017, 'BSI_bands_2017');
var Bands_2018 = AdditionalBands(Combined2018, 'BSI_bands_2018');
var Bands_2019 = AdditionalBands(Combined2019, 'BSI_bands_2019');
var Bands_2020 = AdditionalBands(Combined2020, 'BSI_bands_2020');
var Bands_2021 = AdditionalBands(Combined2021, 'BSI_bands_2021');
var Bands_2022 = AdditionalBands(Combined2022, 'BSI_bands_2022');