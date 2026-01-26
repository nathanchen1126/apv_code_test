
var PV_train_2020 = ee.FeatureCollection("users/liujingxitu/PV/PV_Train_2020");
var NPV_train_2020 = ee.FeatureCollection("users/liujingxitu/PV/NPV_Train_2020");
var PV_test_2020 = ee.FeatureCollection("users/liujingxitu/PV/PV_Test_2020");
var NPV_test_2020 = ee.FeatureCollection("users/liujingxitu/PV/NPV_Test_2020");
var PV_test_2015 = ee.FeatureCollection("users/liujingxitu/PV/PV_Test_2015");
var NPV_test_2015 = ee.FeatureCollection("users/liujingxitu/PV/NPV_Test_2015");
var NL_buffer = ee.FeatureCollection("projects/ee-jinyue627/assets/VIIRS2020");               // nighttime light data derived human area buffer zone
var PV_ESSD = ee.FeatureCollection("projects/ee-jinyue627/assets/PV_China_2020essd");         // existing PV database by Zhang et al. in ESSD

//############################################################################################ 
//####################  The shapefile of each province in China   #######################################          
//############################################################################################ 
var roi_china = ee.FeatureCollection("users/jinyue627/China");
var roi_AnHui     = ee.FeatureCollection("users/liujingxitu/Admin/anhui");                    // Anhui Province
var roi_BJTJ      = ee.FeatureCollection("users/liujingxitu/Admin/BeiJingTianJing");          // Beijing & Tianjin City
var roi_ChongQing = ee.FeatureCollection("users/liujingxitu/Admin/ChongQing");                // Chongqing City
var roi_FuJian   = ee.FeatureCollection("projects/ee-chian/assets/FuJian");                   // Fujian Province
var roi_GDMcHk   = ee.FeatureCollection("projects/ee-chian/assets/MacaoGuandongHongkong");    // Guangdong, HongKong, Macao
var roi_GuangXi  = ee.FeatureCollection("projects/ee-chian/assets/GuangXi");                  // Guangxi Province
var roi_GuiZhou  = ee.FeatureCollection("projects/ee-chian/assets/GuiZhou");                  // Guizhou Province
var roi_HaiNan   = ee.FeatureCollection("projects/ee-chian/assets/HaiNan");                   // Hainan Province
var roi_HeBei   = ee.FeatureCollection("projects/ee-chian/assets/HeiBei");                    // Hebei Province
var roi_HeNan    = ee.FeatureCollection("users/liujingxitu/Admin/HeNan");                     // Henan Province
var roi_HuBei    = ee.FeatureCollection("users/liujingxitu/Admin/HuBei");                     // Hubei Province
var roi_HuNan    = ee.FeatureCollection("users/liujingxitu/Admin/HuNan");                     // Hunan Province
var roi_JiangSu  = ee.FeatureCollection("users/liujingxitu/Admin/JiangSu");                   // Jiangsu Province
var roi_JiangXi  = ee.FeatureCollection("users/liujingxitu/Admin/JiangXi");                   // Jiangxi Province
var roi_JiLin    = ee.FeatureCollection("users/liujingxitu/Admin/JiLin");                     // Jilin Province
var roi_LiaoNing = ee.FeatureCollection("users/liujingxitu/Admin/LiaoNing");                  // Liaoning Province
var roi_NingXia  = ee.FeatureCollection("users/liujingxitu/Admin/NingXia");                   // Ningxia
var roi_ShanDong = ee.FeatureCollection("users/liujingxitu/Admin/ShanDong");                  // Shandong Province
var roi_ShannXi  = ee.FeatureCollection("users/liujingxitu/Admin/ShaanXi");                   // Shaanxi Province
var roi_ShanXi   = ee.FeatureCollection("users/liujingxitu/Admin/ShanXi");                    // Shanxi Province
var roi_ShangHai = ee.FeatureCollection("users/liujingxitu/Admin/ShangHai");                  // ShangHai City
var roi_SiChuan  = ee.FeatureCollection("users/liujingxitu/Admin/SiChuan");                   // Sichuan Province
var roi_TaiWan   = ee.FeatureCollection("users/liujingxitu/Admin/TaiWan");                    // Taiwan Province
var roi_YunNan   = ee.FeatureCollection("users/liujingxitu/Admin/YunNan");                    // Yunnan Province
var roi_Zhejiang = ee.FeatureCollection("users/liujingxitu/Admin/Zhejiang");                  // Zhejiang Province
    
var roi_GS1 = ee.FeatureCollection("users/liujingxitu/Admin/GS1");              // Gansu Province 1
var roi_GS2 = ee.FeatureCollection("users/liujingxitu/Admin/GS2");              // Gansu Province 2
var roi_GS3 = ee.FeatureCollection("users/liujingxitu/Admin/GS3");              // Gansu Province 3
var roi_GS4 = ee.FeatureCollection("users/liujingxitu/Admin/GS4");              // Gansu Province 4
var roi_HLJ1 = ee.FeatureCollection("users/liujingxitu/Admin/HLJ1");            // Heilongjiang Province 1
var roi_HLJ2 = ee.FeatureCollection("users/liujingxitu/Admin/HLJ2");            // Heilongjiang Province 2
var roi_HLJ3 = ee.FeatureCollection("users/liujingxitu/Admin/HLJ3");            // Heilongjiang Province 3
var roi_HLJ4 = ee.FeatureCollection("users/liujingxitu/Admin/HLJ4");            // Heilongjiang Province 4
var roi_NM1 = ee.FeatureCollection("users/liujingxitu/Admin/NM1");              //  Inner Monglia 1
var roi_NM2 = ee.FeatureCollection("users/liujingxitu/Admin/NM2");              //  Inner Monglia 2
var roi_NM3 = ee.FeatureCollection("users/liujingxitu/Admin/NM3");              //  Inner Monglia 3
var roi_NM4 = ee.FeatureCollection("users/liujingxitu/Admin/NM4");              //  Inner Monglia 4
var roi_NM5 = ee.FeatureCollection("users/liujingxitu/Admin/NM5");              //  Inner Monglia 5
var roi_QH1 = ee.FeatureCollection("users/liujingxitu/Admin/QH1");              // Qinghai Province 1
var roi_QH2 = ee.FeatureCollection("users/liujingxitu/Admin/QH2");              // Qinghai Province 2
var roi_QH3 = ee.FeatureCollection("users/liujingxitu/Admin/QH3");              // Qinghai Province 3
var roi_XZ1 = ee.FeatureCollection("users/liujingxitu/Admin/XZ1");              // Tibet 1
var roi_XZ2 = ee.FeatureCollection("users/liujingxitu/Admin/XZ2");              // Tibet 2
var roi_XZ3 = ee.FeatureCollection("users/liujingxitu/Admin/XZ3");              // Tibet 3
var roi_XZ4 = ee.FeatureCollection("users/liujingxitu/Admin/XZ4");              // Tibet 4
var roi_XJ1 = ee.FeatureCollection("users/liujingxitu/Admin/XJ1");              // Xinjiang 1
var roi_XJ2 = ee.FeatureCollection("users/liujingxitu/Admin/XJ2");              // Xinjiang 2
var roi_XJ3 = ee.FeatureCollection("users/liujingxitu/Admin/XJ3");              // Xinjiang 3
var roi_XJ4 = ee.FeatureCollection("users/liujingxitu/Admin/XJ4");              // Xinjiang 4
var roi_XJ5 = ee.FeatureCollection("users/liujingxitu/Admin/XJ5");              // Xinjiang 5
var roi_XJ6 = ee.FeatureCollection("users/liujingxitu/Admin/XJ6");              // Xinjiang 6
var roi_XJ7 = ee.FeatureCollection("users/liujingxitu/Admin/XJ7");              // Xinjiang 7
    

//#################################  User Functions  #########################################        

// L8 mask cloudC01
function maskL8clouds(image) {
  var qa = image.select('pixel_qa');
    var cloudShadowBitMask = 1 << 3;
    var cloudsBitMask = 1 << 5;
    var snowBitMask = 1 << 4;
    var mask = qa.bitwiseAnd(cloudsBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudShadowBitMask).eq(0))
                 .and(qa.bitwiseAnd(snowBitMask).eq(0))
  // return image.updateMask(cloud.not()).updateMask(mask2);
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"]);
}

// L8 mask cloud C02
function maskL8clouds_C02(image) {
  var qa = image.select('QA_PIXEL');
    var cloudShadowBitMask = 1 << 3;
    var cloudsBitMask = 1 << 5;
    var snowBitMask = 1 << 4;
    var mask = qa.bitwiseAnd(cloudsBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudShadowBitMask).eq(0))
                 .and(qa.bitwiseAnd(snowBitMask).eq(0))
  // return image.updateMask(cloud.not()).updateMask(mask2);
  return image.updateMask(mask).multiply(0.0000275).add(-0.2)
      .select("SR_B.*")
      .copyProperties(image, ["system:time_start"]);
}

var applyScaleFactors = function(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}

// A function that scales and masks Landsat 8 (C2) surface reflectance images.
function prepSrL8(image) {
  // Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var getFactorImg = function(factorNames) {
    var factorList = image.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);
  };
  var scaleImg = getFactorImg([
    'REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10']);
  var offsetImg = getFactorImg([
    'REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10']);
  var scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg);

  // Replace original bands with scaled bands and apply masks.
  return image.addBands(scaled, null, true)
    .select("SR_B.*")
    .updateMask(qaMask).updateMask(saturationMask);
}

  
function Get_NDVI(image)
{
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('ndvi');
  return ndvi;
}

function Get_mndwi(image)
{
  var mndwi = image.normalizedDifference(['B3', 'B6']).rename('mndwi');
  return mndwi;
}

function Get_ndbi(image)
{
  var ndbi = image.normalizedDifference(['B6', 'B5']).rename('ndbi');
  return ndbi;
}

function clip_class0(poly)
{
   var img_clip = Class_0.clip(poly);
   return img_clip;
}


//############################################################################################ 
//############################# Start of User control ########################################            
//############################################################################################ 

 
//############################################################################################ 
//#################################  User Controls  ##########################################         
//############################################################################################ 

var cur_ROI = roi_GS4;
var out_name_2020_raw    = 'Vec_GS4_raw_2020';
var out_name_2020_filter = 'Vec_GS4_filter_2020';

var out_name_2015_raw    = 'Vec_GS4_raw_2015';
var out_name_2015_filter = 'Vec_GS4_filter_2015';

//var proj_cur    = 'EPSG:32650';     // select the UTM for the input ROI province 
var proj_cur    = 'EPSG:32647';

var start_L8_2015    = "2015-10-01" ;   // try to select the latest months which have a whole image composite witthout cloud
var finish_L8_2015   = "2015-12-31";
var start_L8_2020    = "2020-10-01"  ;  
var finish_L8_2020   = "2020-12-31";

var user_L8_reso = 30;  
var user_tile    = 16;  // 1 to 16

var user_limit_pixelcount = 9;
var user_glcm_size = 5;      
var user_cloud2  = 10;  
var user_cloud   = 10 ;    



var flag_export_previous_PV_stats = 0;
var flag_export_sammples = 0;
var flag_showConMat_raw = 0;
var flag_showConMat_post = 0;
var flag_showImChart = 0;
var flag_samples_known = 0;



//########################
var VisRGB      = { min: 0, max: 0.25, bands: ["B4", "B3", "B2"]};  
var VisClassify = { min: 0, max: 1, palette: ['black', 'white']};
var VisAspect   = { min: 0, max: 359.99, palette: ['navy','blue','green','yellow','white','pink','red','brown']};
var VisSlope    = { min: 0, max: 59.99, palette: ['white','grey','lightblue', 'green','blue','purple','brown']};
var VisVector   = { color: 'yellow' };
var VisNDVI    = { min: 0, max: 0.8, palette: ['white','lightyellow', 'yellow', 'green', 'blue','purple']};
var VismNDWI    = { min: -0.4, max: 0.5, palette: ['pink','lightpink', 'white','lightyellow', 'yellow', 'green', 'blue','purple']};
var VisVectorESSD  = {'color': 'blue', 'width': 3, 'fillColor': '00000000'};
var VisVectorOut = { 'color': 'pink', 'width': 3, 'fillColor': '00000000'};
var VisVectorROI = { 'color': 'black', 'width': 3, 'fillColor': '00000000'};

var user_bands_TextureL8  = ['B1','B2','B3','B4','B5','B6','B7'];
var user_bands_TextureIdx = ["ndbi", "mndwi", "ndvi"];
var user_bands_TextureDEM = ["aspect","slope"];


var user_bands_classify = ['B1','B2','B3','B4','B5','B6','B7',"mndwi","ndbi","ndvi"]
var user_bands_classify = ['B1','B2','B3','B4','B5','B6','B7' ,"mndwi","ndbi","ndvi",
                           'B2_savg',  'B6_savg', 'B7_savg', 
                           'ndbi_savg', 'mndwi_savg', 'ndvi_savg'];

                            
//########################################################################################## 
//############################# End of User control ########################################            
//########################################################################################## 
//var training = NPV.merge(PV1); 
var training_2020 = PV_train_2020.merge(NPV_train_2020); 
var test_2020 = PV_test_2020.merge(NPV_test_2020); 



//##############################################################################
//############################   Getting images ready   ########################
//##############################################################################


//var L8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")   //used by Wangjinyue
var L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"); 

var L8Image = L8.filterDate(start_L8_2020, finish_L8_2020)
                //.map(applyScaleFactors)
                .map(maskL8clouds_C02)
                //.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',user_cloud))
                //.map(prepSrL8)
                .median(); 
            
var L8Image = L8Image.rename(['B1','B2','B3','B4','B5','B6','B7']);
print(L8Image,"L8Image");
Map.addLayer(L8Image, VisRGB, 'L8global_2020');


var NasaDEMdata = ee.Image('NASA/NASADEM_HGT/001');
var DEM = NasaDEMdata.select('elevation');
var Slope = ee.Terrain.slope(DEM);
var Hillshade = ee.Terrain.hillshade(DEM);
var Aspect = ee.Terrain.aspect(DEM);           // Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W.
var DEMImage = Aspect.addBands(Slope);
//Map.addLayer(Slope, VisSlope, 'Slope');
//Map.addLayer(Aspect, VisAspect, 'Aspect');


//############ get Max NDVI ##############
var L8_list =  L8.filterDate(start_L8_2020, finish_L8_2020)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',user_cloud2))
                  .map(maskL8clouds);

var ndvi_list  = L8_list.map(Get_NDVI); 
var ndvi_max   = ndvi_list.max();
var mndwi_list = L8_list.map(Get_mndwi); 
var mndwi_max  = mndwi_list.max();
var ndbi_list  = L8_list.map(Get_ndbi); 
var ndbi_max   = ndbi_list.max();

var ndbi = L8Image.normalizedDifference(['B6', 'B5']).rename(['ndbi']);
var ndvi = L8Image.normalizedDifference(['B5', 'B4']).rename(['ndvi']);
var mndwi = L8Image.normalizedDifference(['B3', 'B6']).rename(['mndwi']);


// var IdxImage = ndbi.addBands(mndwi_max).addBands(ndvi_max) ;   //higher accuracy than using ndvi from L8composite, ndbi > ndbi_max, mndwi>ndwi
var IdxImage = ndbi.addBands(mndwi).addBands(ndvi);   
//Map.addLayer(ndvi_max,VisNDVI,'ndvi_max' )
//Map.addLayer(mndwi_max,VismNDWI,'mndwi');
//Map.addLayer(ndbi_max,VismNDWI,'ndbi_max');
//Map.addLayer(ndbi,VismNDWI,'ndbi');


var ivL8 = L8Image.select(user_bands_TextureL8).add(127.5).multiply(127.5).toUint16();
var ivIdx = IdxImage.select(user_bands_TextureIdx).add(127.5).multiply(127.5).toUint16();
var ivDEM = DEMImage.select(user_bands_TextureDEM).add(127.5).multiply(127.5).toUint16();

var glcmL8 = ivL8.glcmTexture(user_glcm_size);
var glcmIdx = ivIdx.glcmTexture(user_glcm_size);
var glcmDEM = ivDEM.glcmTexture(user_glcm_size);

var Img_classify = L8Image.addBands(IdxImage).addBands(DEMImage)
                  .addBands(glcmL8).addBands(glcmIdx).addBands(glcmDEM);
                  
print('Img_classify ready for classification in 2020: ');
print(Img_classify);
 


//##############################################################################
//########################## Training & get classifier   #######################
//##############################################################################
print('Chosen features for classification : ');
print(user_bands_classify);

var TrainShp = training_2020;
var TrainFea = Img_classify.select(user_bands_classify).sampleRegions({
    collection: TrainShp,
    properties: ['landcover'],
    projection: proj_cur,
    scale: user_L8_reso,
    tileScale: user_tile }); 
    
//print('training Features: ');
//print(TrainFea.first());

print('start training random forest....');
var RF_trained = ee.Classifier.smileRandomForest({numberOfTrees:100}).train({
  features: TrainFea,
  classProperty: 'landcover',
  inputProperties: user_bands_classify
});


//##############################################################################
//########################   Classify the test samples.     ####################
//##############################################################################

var TestShp = test_2020;
//print('test Features: ');
//print(TestShp.first());

if (flag_samples_known === 0)
{
    print('starting getting feature values for test samples.......');
    //print('user_bands_classify :')
    //print(user_bands_classify)
    var TestFea = Img_classify.select(user_bands_classify).sampleRegions({
    collection: TestShp,
    properties: ['landcover'],
    projection: proj_cur,
    scale: user_L8_reso,
    tileScale: user_tile,
    geometries: true});
}
else
{
    print('Input test samples has feature values ---->');
    var TestFea = input_Test;
}
var validated_RF = TestFea.classify(RF_trained);
//print('validated_RF: ');
//print(validated_RF.limit(2));

//##############################################################################
//####################      Accuracy evaluation       ##########################
//##############################################################################

if (flag_showConMat_post == 1)
{
  var ConMat = validated_RF_opened.errorMatrix('landcover', 'opened');    //## Accuracy evaluation After Post Processing 
  print('ConfusionMatrix after PostProcessing..........',ConMat);
  print('Overall accuracy', ConMat.accuracy());
  print('Kappa accuracy', ConMat.kappa());
  print("Consumer's accuracy", ConMat.consumersAccuracy());
  print("Producer's accuracy", ConMat.producersAccuracy());
}


//##############################################################################
//####################    Feature Importance from RF    ########################
//##############################################################################
var dict = RF_trained.explain();
var FeImport = ee.Feature(null, ee.Dictionary(dict).get('importance'));
var ImChart = ui.Chart.feature.byProperty(FeImport)
              .setChartType('ColumnChart')
              .setOptions({
                   title: 'Random Forest Variable Importance',
                   legend: {position: 'none'},
                   hAxis: {title: 'Bands'},
                   vAxis: {title: 'Importance'}
              });
//print(ImChart);
if (flag_showImChart == 1) {print(ImChart);}

if (flag_showConMat_raw == 1)
{
  var ConMat = validated_RF.errorMatrix('landcover', 'classification');
  print('ConfusionMatrix before PostProcessing..........',ConMat);
  print('Overall accuracy', ConMat.accuracy());
  print('Kappa accuracy', ConMat.kappa());
  print("Consumer's accuracy", ConMat.consumersAccuracy());
  print("Producer's accuracy", ConMat.producersAccuracy());
}


//########################################################################################
//######################  Classify on Image  and Post Processing  ########################
//########################################################################################

var Class0_2020 = Img_classify.select(user_bands_classify).classify(RF_trained);

var kernel = ee.Kernel.circle({radius: 1});
var kernel = ee.Kernel.square({radius: 1});
var Opened_2020 = Class0_2020
             .focalMin({kernel: kernel, iterations: 1})
             .focalMax({kernel: kernel, iterations: 1});
print('Opening done ...... ');
var Opened_2020 = Opened_2020.select(['classification']).rename(['opened']);

var validated_RF_opened = Opened_2020.select('opened').sampleRegions({
  collection: validated_RF,  
  properties: ['classification', 'landcover'],
  //projection: proj_ZJ,
  scale: user_L8_reso,
  tileScale: user_tile,
});

//print('validated_RF_opened: ');
//print(validated_RF_opened.limit(2));



//##############################################################################
//############################  Display results   ##############################
//##############################################################################

var Class0      = Class0_2020.clip(roi_china);
var Opened_2020 = Opened_2020.clip(roi_china);
var After_light_2020 = Opened_2020.clip(NL_buffer);
var L8 = L8Image.clip(roi_china);
 
Map.addLayer(Class0_2020, VisClassify, 'Class0_2020');
Map.addLayer(Opened_2020, VisClassify, 'Opened_2020');
Map.addLayer(After_light_2020, VisClassify, 'After_light_2020');

//###########################################################################################
//############################  Raster to Vector, and Export to drive  ######################
//###########################################################################################
var Vec_2020_raw = Class0.reduceToVectors({
   reducer:ee.Reducer.countEvery(),
   geometry: cur_ROI,
   //geometry: temp_region,
   scale: user_L8_reso,
   geometryType:"polygon",
   maxPixels:1e20,
   tileScale:16
 });
var Vec_2020_out_raw = Vec_2020_raw.filter(ee.Filter.eq('label', 1)); 
Map.addLayer(Vec_2020_raw.style(VisVectorOut),{}, 'Vec_2020_raw');  

var Vec_2020_filter = After_light_2020.reduceToVectors({
   reducer:ee.Reducer.countEvery(),
   geometry: cur_ROI,
   //geometry: temp_region,
   scale: user_L8_reso,
   geometryType:"polygon",
   maxPixels:1e20,
   tileScale:16
});
         
var Vec_2020_out_filter = Vec_2020_filter.filter(ee.Filter.eq('label', 1));                   //only keep PV detection 
var Vec_2020_out_filter = Vec_2020_out_filter.filter(ee.Filter.gt('count', user_limit_pixelcount)); 
Map.addLayer(Vec_2020_out_filter.style(VisVectorOut),{}, 'Vec_2020_out_filter');


Export.table.toDrive({
  collection: Vec_2020_out_raw ,
  description: out_name_2020_raw,
  fileFormat: 'shp',
  folder: 'PV_2020_shp',
});   
Export.table.toDrive({
  collection: Vec_2020_out_filter ,
  description: out_name_2020_filter,
  fileFormat: 'shp',
  folder: 'PV_2020_shp',
});   




//###########################################################################################
//############################  Set the location to zoom to   ###############################
//###########################################################################################
        
Map.addLayer(cur_ROI.style(VisVectorROI),{}, 'cur_ROI'); 
Map.setCenter(119.25, 32.96, 14);
Map.setCenter(100.5664, 36.1169, 12);
Map.setCenter(82.3332, 44.6495, 11);
Map.setCenter(94.59, 40.90, 11);
Map.setCenter(86.1667, 42.10339, 13);
        
Map.setCenter(100.59,  36.17, 12);  
Map.setCenter(98.97585, 39.96789, 14);     // ##### Gansu JiuQuang
               
         
      
    

   
//########################################################################################
//############################  Apply the classifer to 2015 ##############################
//######################################################################################## 


var L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"); 
var L8Image_2015 = L8.filterDate(start_L8_2015, finish_L8_2015)
                   .map(maskL8clouds_C02)
                   .median(); 
var L8Image_2015 = L8Image_2015.rename(['B1','B2','B3','B4','B5','B6','B7'])
print(L8Image_2015,"L8Image_2015")            
Map.addLayer(L8Image_2015, VisRGB, 'L8_2015');              

var ndbi = L8Image_2015.normalizedDifference(['B6', 'B5']).rename(['ndbi']);
var ndvi = L8Image_2015.normalizedDifference(['B5', 'B4']).rename(['ndvi']);
var mndwi = L8Image_2015.normalizedDifference(['B3', 'B6']).rename(['mndwi']);

var IdxImage_2015 = ndbi.addBands(mndwi).addBands(ndvi);   
var Img_classify_2015 = L8Image_2015.addBands(IdxImage_2015)
                   .addBands(DEMImage)
                   .addBands(glcmL8).addBands(glcmIdx).addBands(glcmDEM);
print('Img_classify ready for classification : ');
print(Img_classify_2015);


var Class0_2015 = Img_classify_2015.select(user_bands_classify).classify(RF_trained);
Map.addLayer(Class0_2015, VisClassify, 'RawPV_2015');

var Class0_2015 = Class0_2015.clip(roi_china);
var kernel = ee.Kernel.square({radius: 1});

// Perform an erosion followed by a dilation, display.
var opened_2015 = Class0_2015
             .focalMin({kernel: kernel, iterations: 1})
             .focalMax({kernel: kernel, iterations: 1});
print('Opening done ...... ');
var Opened_2015 = opened_2015.select(['classification']).rename(['opened']);


//var opened_2015 = opened_2015 .clip(cur_ROI);
var After_light_2015 = Opened_2015.clip(NL_buffer);
//var L8_2015_curROI   = L8Image_2015.clip(cur_ROI);
//Map.addLayer(L8_2015_curROI, VisRGB, 'L8_2015_curROI');
Map.addLayer(Opened_2015, VisClassify, 'Opened_2015');
Map.addLayer(After_light_2015, VisClassify, 'After_light_2015');


//###########################################################################################
//############################  Raster to Vector, and Export to drive  ######################
//###########################################################################################
var Vec_2015_raw = Class0_2015.reduceToVectors({
   reducer:ee.Reducer.countEvery(),
   geometry: cur_ROI,
   //geometry: temp_region,
   scale: user_L8_reso,
   geometryType:"polygon",
   maxPixels:1e20,
   tileScale:16
 });
var Vec_2015_out_raw = Vec_2015_raw.filter(ee.Filter.eq('label', 1)); 
Map.addLayer(Vec_2015_raw.style(VisVectorOut),{}, 'Vec_2015_raw');  

var Vec_2015_filter = After_light_2015.reduceToVectors({
   reducer:ee.Reducer.countEvery(),
   geometry: cur_ROI,
   //geometry: temp_region,
   scale: user_L8_reso,
   geometryType:"polygon",
   maxPixels:1e20,
   tileScale:16
});
         
var Vec_2015_out_filter = Vec_2015_filter.filter(ee.Filter.eq('label', 1));                   //only keep PV detection 
var Vec_2015_out_filter = Vec_2015_out_filter.filter(ee.Filter.gt('count', user_limit_pixelcount)); 
Map.addLayer(Vec_2015_out_filter.style(VisVectorOut),{}, 'Vec_2015_out_filter');


Export.table.toDrive({
  collection: Vec_2015_out_raw ,
  description: out_name_2015_raw,
  fileFormat: 'shp',
  folder: 'PV_2015_shp',
});   
Export.table.toDrive({
  collection: Vec_2015_out_filter ,
  description: out_name_2015_filter,
  fileFormat: 'shp',
  folder: 'PV_2015_shp',
});   


Map.addLayer(PV_ESSD.style(VisVectorESSD),{}, 'PV_ESSD'); 
 
 
 

