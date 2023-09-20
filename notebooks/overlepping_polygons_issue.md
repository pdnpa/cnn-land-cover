## Overlapping polygon issue

* content\evaluation_polygons\landscape_character_2022_detailed_CFGH-override

When merging the OS NGD and NFI data with mainclass predictions this was done using vector data overlayed on raster data. As such, surrounding the CFGH overide layer, there are many pixels that have a null value. To fix this in GIS the following method is employed susccesfully. 

**QGIS 3.10 & ARC GIS pro 3.1.3 used in this process**

1) Topology Checker **Version 0.1** 
2) V.clean **(https://grass.osgeo.org/grass82/manuals/v.clean.html)[v.clean]**
3) Dissolve via 'lc_class' column (removes all those new overlap polygons as they dissolve in to their neighbor polygons with the same 'lc_class') **(https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/dissolve.htm)[dissolve (Data Management)]**
4) Multipart to singlepart (restores all the segments as individual polygons again!) **(https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/multipart-to-singlepart.htm)[Multipart To Singlepart (Data Management)]**

In future, the ordering will change so the CFGH overide layer is processed once the model predictions have been vectorised