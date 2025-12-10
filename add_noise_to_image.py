from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the PNG image
img = Image.open("start_image_no_noise.png")

# Convert to grayscale
img_gray = img.convert("L")   # "L" mode = 8-bit grayscale

# Convert to NumPy array (2D)
array_of_image = np.array(img_gray)

def show_image(image_array, title="Image"):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

print(array_of_image.shape)
#show_image(array_of_image, title="Original Grayscale Image")


def sigma_k_from_target_snr(img, target_snr_db):
    M, N = img.shape
    mn = float(M * N) 
    mean_sig2 = np.mean(img.astype(np.float64)**2)
    snr_lin = 10.0**(target_snr_db / 10.0)
    #print("mean signal^2:", mean_sig2)
    sigma_img_sq = mean_sig2 / snr_lin
    sigma_k = np.sqrt(mn * sigma_img_sq)
    #print(sigma_k)
    return sigma_k
'''
SNRlin​=σimg2​mean(signal2)​
it makes sense to consider nl itself
stationary, which implies that σ2
Kl
(k) = σ2
Kl 
is a constant, and therefore the variance
of noise does not depend on the position. (pg 32)
'''



img = Image.open("start_image_no_noise.png").convert("L")
img = np.array(img, dtype=np.float32)


kspace = np.fft.fftshift(np.fft.fft2(img))

# ------------------------------------------------------------
#  Add complex AWGN noise in k-space
#    sl(k) = al(k) + nl(k; 0, σ²)
# ------------------------------------------------------------

#sigma = 0.001 * np.max(np.abs(kspace))   # choose noise level as % of signal magnitude
sigma= sigma_k_from_target_snr(img, target_snr_db=40)  # target SNR in dB

noise_real = np.random.normal(0, sigma**2, kspace.shape)
noise_imag = np.random.normal(0, sigma**2, kspace.shape)#just pretend its imaginary !!!!
j_to_match_equations=1# this is j, j is supposed to be sqrt(-1) but  as im modeling the iamginary noise as a real - value i dont have to bother witht this

noise = noise_real + 1 * noise_imag #measuring the magnitude of the noise, means that we are adding both real and imaginary parts (I know we could just double it but its not the same thing in principle even though it is in practice at the moemnt)

kspace_noisy = kspace +  noise # adding noise to k-space modifier to control noise level while i work out other stuff


img_noisy = np.fft.ifft2(np.fft.ifftshift(kspace_noisy))
img_noisy = np.abs(img_noisy)   # magnitude reconstruction (for single-coil)


show_image(img_noisy, title="Grayscale Image in ")
