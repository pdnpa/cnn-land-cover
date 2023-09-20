## Overlapping polygon issue

* content\evaluation_polygons\landscape_character_2022_detailed_CFGH-override

Merging the OS NGD and NFI layer with mainclass prediction layer was done using NGD/NFI vector data overlayed on (raster-)pixelated prediction vector data. As a result, surrounding the CFGH overide layer, there are many (sub) pixels that have a null value, or overlap between NGD/NFI and prediction polgons. This created a 'saw teeth' effect along the CFGH overide polygons. To fix this in GIS the following method is employed succesfully. 

*QGIS 3.10 & ARC GIS pro 3.1.3 used in this process*

1) Topology Checker [Version 0.1](https://docs.qgis.org/3.28/en/docs/user_manual/plugins/core_plugins/plugins_topology_checker.html) 
2) V.clean [v.clean](https://grass.osgeo.org/grass82/manuals/v.clean.html)
3) Dissolve via 'lc_class' column (removes all those new overlap polygons as they dissolve in to their neighbour  polygons with the same 'lc_class') [dissolve (Data Management)](https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/dissolve.htm)
4) Multipart to singlepart (restores all the segments as individual polygons again!) [Multipart To Singlepart (Data Management)](https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/multipart-to-singlepart.htm)

In future, the ordering will change so the CFGH override layer is processed once the model predictions have been vectorised.
