
name = sys.argv[1]
image = cv2.imread(name, 0)

converter = MaskToPolygons()
categoryToPolygons, debug = converter.processImage(image)
plt.imshow(debug)
plt.show()

places2_path = "/data/vision/oliva/scenedataset/places2new/challenge2016/data_large"
    # Symlink images
    original_images_path = "{}/{}/{}".format(places2_path, category[0], category)
    os.symlink(original_images_path, images_path)