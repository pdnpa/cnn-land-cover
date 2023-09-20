## Overlapping polygon issue

* content\evaluation_polygons\landscape_character_2022_detailed_CFGH-override

Merging the OS NGD and NFI layer with mainclass prediction layer was done using NGD/NFI vector data overlayed on (raster-)pixelated prediction vector data. As a result, surrounding the NGD/NFI overide layer, there are many (sub) pixels that have a null value, or overlap between NGD/NFI and prediction polgons. This created a 'saw teeth' effect along the CFGH overide polygons. To fix this in GIS the following method is employed succesfully. 

*QGIS 3.10 & ARC GIS pro 3.1.3 used in this process*

1) Buffer of 20cm ground resolution (ensures all missing pixels are covered byNGD/NFI vector data) [buffer](https://docs.qgis.org/3.28/en/docs/gentle_gis_introduction/vector_spatial_analysis_buffers.html)
2) Topology Checker (searches for overlaps created by buffer edges) [version 0.1](https://docs.qgis.org/3.28/en/docs/user_manual/plugins/core_plugins/plugins_topology_checker.html) 
3) V.clean (cleans overlaps of multiple polygons into single (layer) polygons) [v.clean](https://grass.osgeo.org/grass82/manuals/v.clean.html)
4) Dissolve via 'lc_class' column (removes new single overlap polygons as they dissolve in to their neighbour polygons with the same 'lc_class') [dissolve (Data Management)](https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/dissolve.htm)
5) Multipart to singlepart (restores all the segments as individual polygons again!) [multipart to singlepart (Data Management)](https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/multipart-to-singlepart.htm)
6) Using the resulting file from the 5 steps above to create difference maps to then merge the new NGD/NFI polygons and the predictions [difference](https://docs.qgis.org/3.28/en/docs/user_manual/processing_algs/qgis/vectoroverlay.html#difference)


In future, the ordering will change so the CFGH override layer is processed once the model predictions have been vectorised.
