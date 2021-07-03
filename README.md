# Boat Detection

Automated analysis of traffic scenes mainly focuses on road scenes. However, similar methods can also be applied in a 
different scenario: boat traffic in the sea. The goal of this project is to develop a system capable of detecting boats in the image.
You should consider that the sea surface is not regular, and it is moving (and could lead to white trails), and the 
appearance of a boat can change sensibly from model to model, and depending on the viewpoint. Boat detection can be 
tackled in several different ways:the ultimate goal is to locate the boat in the image (e.g., drawing a rectangle around 
the boat). Consider that the boat can appear at any scale. For boat detection, the Intersection over Union (IoU) metric
can be exploited for the performances evaluation (https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection).
