import h5py
import numpy as np
import matplotlib.pyplot as plt

def flipboook(magnitude_images):
    for i in range(magnitude_images.shape[0]):
        plt.imshow(magnitude_images[i], cmap='gray')
        plt.title(f"Slice {i}")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.2)  # 0.2 seconds between slices
        plt.clf()

def show_dataset(dataset, key_name):
    data = dataset[:]
    print(key_name, data.shape, data.dtype)
    
    # Handle complex k-space: take magnitude after IFFT if 2D or 3D
    if np.iscomplexobj(data):
        # For 2D
        if data.ndim == 2:
            print(f"{key_name} has 2 dimensions, displaying image. A")
            image = np.fft.ifft2(data)
            magnitude = np.abs(np.fft.fftshift(image))
            plt.imshow(magnitude, cmap='gray')
            plt.title(key_name)
            plt.show()
        # For 3D (multiple slices)
        elif data.ndim == 3:
            print(f"{key_name} has 3 dimensions, displaying middle slice. B")
            lengthData = data.shape[0]
            all_images=[]
            for i in range(lengthData):
                image = np.fft.ifft2(data[i])
                magnitude = np.abs(np.fft.fftshift(image))
                all_images.append(magnitude)
            flipboook(np.array(all_images))
            #image = np.fft.ifft2(data[mid])
            #magnitude = np.abs(np.fft.fftshift(image))
            #plt.imshow(magnitude, cmap='gray')
            #plt.title(f"{key_name} (slice {mid})")
            #plt.show()
    # If real-valued data
    else:
        if data.ndim == 2:
            print(f"{key_name} has 2 dimensions, displaying image. C")
            plt.imshow(data, cmap='gray')
            plt.title(key_name)
            plt.show()
        elif data.ndim == 3:
            print(f"{key_name} has 3 dimensions, displaying middle slice. D")
            mid = data.shape[0] // 2
            plt.imshow(data[mid], cmap='gray')
            plt.title(f"{key_name} (slice {mid})")
            plt.show()

def see_real_knee_mri_with_noise():
    with h5py.File("real_Knee_MRI_has_noise.h5", "r") as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset) and item.shape != ():  # scalar check
                show_dataset(item, key)
            else:
                print(f"{key} is not a sliceable dataset, skipping...")

see_real_knee_mri_with_noise()

