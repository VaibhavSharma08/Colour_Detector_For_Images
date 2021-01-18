from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from collections import Counter


def returnHexNumbers(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


def inputImage(imagePath):
    image = cv2.imread(imagePath)

    # Changing Image size for reducing pixels which in turn reduces the time taken
    decreasedImage = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)

    # Converting to RGB since OpenCV stores in BGR by default
    decreasedImage = cv2.cvtColor(decreasedImage, cv2.COLOR_BGR2RGB)

    # Showing initial Image
    cv2.imshow("Image for Colour Detection", decreasedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return decreasedImage


def showDetectedColours(decreasedImage, coloursToShow):

    # Changing the shape of the Image object according to the format needed for K mean clustering
    adjustedImage = decreasedImage.reshape(decreasedImage.shape[0] * decreasedImage.shape[1], 3)

    # Implementing K Means clustering
    clf = KMeans(n_clusters=coloursToShow)
    labels = clf.fit_predict(adjustedImage)

    # Storing and Sorting the labels in decreasing order according to their percentages
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [returnHexNumbers(ordered_colors[i]) for i in counts.keys()]

    # Showing the pie chart containing the results
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.show()


if __name__ == '__main__':
    showDetectedColours(inputImage('image.jpg'), 8)
