# sequential-connected-component-labeling-c-and-python

Introduction:
-----------------
Connected component labeling (CCL) is a foundation to early computer vision. Most models implement a two-pass method for assigning individual pixel to a component within the image. Labeling can be achieved object by analyzing 4 or 8 connected neighbors.

Technologies Used:
-----------------
- Python 3.6
- C++

What the Model Does:
------------------ 
This algorithm takes two inputs:
    1. Image B
    2. Pixel size threshold
    
From these values it generates:
    1. all component's size
    2. location of each component's centroid
    3. the bounding box of each object
    4. the orientation of the axis of elongation
    5. the eccentricity, perimeter and compactness 


The image input is taken as a gray level .img file then it is thresholded resulting in the binary image below:
[!](https://i.ibb.co/TPmz6tj/og.png)
