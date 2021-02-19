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

    3. Pixel size threshold
    
From these values it generates:

    1. all component's size
    
    3. location of each component's centroid
    
    4. the bounding box of each object
   
    5. the orientation of the axis of elongation
    
    6. the eccentricity, perimeter and compactness 


The image input is taken as a gray level .img file then it is thresholded resulting in the binary image below:

![](https://i.ibb.co/TPmz6tj/og.png)

Pixels are then raster scanned in two passes labeling connected pixels of each object. The 4-connected neighbor method analyzes only the upper and left pixel as in scans the image, resulting in the next image:


![](https://i.ibb.co/QYmcnfZ/out.png)

However, this is not the end of information that can be mined from a simple raster scan. Next, we can collect values like area, first area and second area moments. [Here is a good source for the theory.](http://www.cse.msu.edu/~stockman/Book/ch3.pdf)


![](https://i.ibb.co/1z7wZsH/eq.png)


Second moments allow you to understand object aspects such as elongation, eccentricity and axis of orientation of the object (AKA axis with least second moment). 


![](https://i.ibb.co/5nZgxWt/tan.png)


Then features like compactness, perimeter and bound boxes can be generated. This deploys a "Moore neighborhood" implementation that sums the lengths of 4 and 8 neighbors around an object in a count-clockwise manner. See the image below for how the primeterfuction works and how BBox labeling is conducted.


![](https://media.giphy.com/media/2k37U1Hjl5RnHScLEe/giphy.gif)


![](https://media.giphy.com/media/oyzMey5mSM7Q49RhDw/giphy.gif)


