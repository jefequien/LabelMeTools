import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

class MaskToPolygons:

    def __init__(self):
        pass

    def processImage(self, image):
        categoryToMask = self.createMasks(image)
        categoryToPolygons = {}

        debug = np.zeros(image.shape)

        for category in categoryToMask.keys():
            mask = categoryToMask[category]
            mask = mask.astype(np.uint8)

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c,h in zip(contours, hierarchy[0]) if cv2.contourArea(c) > 100 and h[3] < 0]

            # Visualize
            for c in contours:
                color = np.random.randint(0, 255, size=3)
                cv2.drawContours(debug, [c], -1, (color[0], color[1], color[2]), 3)

            polygons = []
            for c in contours:
                n = c.shape[0]
                c = c.reshape((n,2))
                polygon = c.tolist()
                polygons.append(polygon)

            categoryToPolygons[category] = polygons

        return categoryToPolygons, debug

    def createMasks(self, image):
        categoryToMask = {}
        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                c = image[i][j]
                if c not in categoryToMask:
                    new_mask = np.zeros(image.shape)
                    categoryToMask[c] = new_mask
                categoryToMask[c][i][j] = 1
        return categoryToMask

if __name__=="__main__":
    name = sys.argv[1]
    image = cv2.imread(name, 0)

    converter = MaskToPolygons()
    categoryToPolygons, debug = converter.processImage(image)
    plt.imshow(debug)
    plt.show()

