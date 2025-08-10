import numpy as np
from matplotlib import pyplot as plt

file_path = "./full_numpy_bitmap_bicycle.npy"
image = np.load(file_path).astype(np.float32)
test_image = image[100]

# categories = ['bicycle', 'apple', 'The Eiffel Tower', 'The Mona Lisa']
categories = ['bicycle']
scores = []
weight = []
for category in categories:
    file_path = f"./full_numpy_bitmap_{category}.npy"
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis=0)
    weight.append(avg_image)
    scores.append(test_image @ avg_image)

print(scores)

print(f'the test_image is most likely {categories[np.argmax(scores)]}')

plt.figure(figsize=(10, 4))
for i in range(len(weight)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight[i].reshape(28, 28))
    plt.axis('off')
    plt.title(categories[i])
plt.show()
