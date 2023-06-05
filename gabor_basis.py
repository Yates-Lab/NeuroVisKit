#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
%matplotlib inline
def oddify(i):
    return i + 1 - i % 2
def locality(x):
    # calculates the locality of the energy of a batch of images in shape b, nx, ny
    tx, ty = torch.meshgrid(torch.linspace(-1, 1, oddify(x.shape[1])), torch.linspace(-1, 1, oddify(x.shape[2])))
    locality_kernel = torch.sqrt(tx**2 + ty**2).to(x.device).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x.unsqueeze(1)**2, locality_kernel, padding="valid").mean()

image = torch.randn(1, 35, 35, requires_grad=True)
dotted = torch.zeros((1, 35, 35))
dotted[:, 10:25, 10:25] = 1
for i in range(100):
    loss = locality(image) #- torch.abs(image*dotted).sum()
    loss.backward()
    image.data -= 0.1*image.grad
    image.grad.zero_()
plt.figure()
plt.imshow(image.squeeze().detach())
#%%
def gabor_spectra(wx, wy, wt, fx, fy, ft, sx, sy, st, p):
    pt1 = np.exp(-1j*p-2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return pt1-pt2

nx, ny, nt = 35, 35, 24
fx, fy, ft = np.meshgrid(
    np.fft.fftshift(np.fft.fftfreq(nx, 1/nx)),
    np.fft.fftshift(np.fft.fftfreq(ny, 1/ny)),
    np.fft.fftshift(np.fft.fftfreq(nt, 1/nt))
)
x, y, t = np.meshgrid(
    np.linspace(0, 1, nx),
    np.linspace(0, 1, ny),
    np.linspace(0, 1, nt)
)
r = np.sqrt(fx**2 + fy**2 + ft**2)
rmod = r[nx//2, ny//2:, nt//2]
fcr, srs, spacings, nangles = [0], [1], [1.5], [1]
nextI = 0
while nextI < max(nx, ny, nt)//2:
    nextI = nextI + spacings[-1]
    srs.append(2/nextI)
    spacings.append(1.3/srs[-1])
    nangles.append(np.pi/np.arctan(spacings[-1]/2/nextI))
    fcr.append(nextI)
    
def hanning(r, n):
    return 0.5+0.5*np.cos(np.pi*r/(n//2))

gabor_ft = np.zeros((nx, ny))
params = []
p = np.pi/2
picked_units = np.arange(len(fcr))#[0] + [i for i in range(1, len(fcr)) if spacings[i]/1.3 > 1]
print("Picked units:", picked_units)
for i in picked_units:
    nspacings = 2*np.pi/nangles[i]
    phis = np.arange(nspacings/2*int(i%2==1), np.pi + nspacings/2*int(i%2==1), nspacings)
    for phiI, phi in enumerate(phis):
        thetas = np.arange(nspacings/2*int(i%2!=phiI%2), 2*np.pi + nspacings/2*int(i%2!=phiI%2), nspacings)
        for theta in thetas:
            fcx, fcy, fct = fcr[i]*np.cos(theta)*np.sin(phi), fcr[i]*np.sin(theta)*np.sin(phi), fcr[i]*np.cos(phi)
            sr = srs[i]
            if abs(fcx) + spacings[i]/2 < nx//2 and abs(fcy) + spacings[i]/2 < ny//2 and abs(fct) + spacings[i]/2 < nt//2:
                params.append([fcx, fcy, fct, sr, sr, sr, p])
                temp_ft = gabor_spectra(fx, fy, ft, fcx, fcy, fct, sr, sr, sr, p)
                gabor_ft += np.abs(temp_ft[:, :, nt//2])
            else:
                print("Skipped:", spacings[i], fcr[i])

params = np.array(params)
np.save("gabor_params.npy", params)
plt.figure()
plt.imshow(gabor_ft)

#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
%matplotlib inline

def gabor_spectra(wx, wy, fx, fy, sx, sy, p):
    pt1 = np.exp(-1j*p-2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2))
    pt2 = np.exp(1j*p-2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2))
    return pt1-pt2

nx, ny = 35, 35
fx, fy = np.meshgrid(
    np.fft.fftshift(np.fft.fftfreq(nx, 1/nx)),
    np.fft.fftshift(np.fft.fftfreq(ny, 1/ny)),
)
x, y = np.meshgrid(
    np.linspace(0, 1, nx),
    np.linspace(0, 1, ny),
)
r = np.sqrt(fx**2 + fy**2)
rmod = r[nx//2, ny//2:]
fcr, srs, spacings, nangles = [0], [1], [1.5], [1]
nextI = 0
while nextI < max(nx, ny)//2:
    nextI = nextI + spacings[-1]
    srs.append(2/nextI)
    spacings.append(1.3/srs[-1])
    nangles.append(np.pi/np.arctan(spacings[-1]/2/nextI))
    fcr.append(nextI)
    
def hanning(r, n):
    return 0.5+0.5*np.cos(np.pi*r/(n//2))

gabor_ft = np.zeros((nx, ny))
params = []
p = np.pi/2
picked_units = np.arange(len(fcr))#[0] + [i for i in range(1, len(fcr)) if spacings[i]/1.3 > 1]
print("Picked units:", picked_units)
for i in picked_units:
    nspacings = 2*np.pi/nangles[i]
    thetas = np.arange(nspacings/2*(i%2), 2*np.pi + nspacings/2*(i%2), nspacings)
    for theta in thetas:
        fcx, fcy = fcr[i]*np.cos(theta), fcr[i]*np.sin(theta)
        sr = srs[i]
        if abs(fcx) + spacings[i]/2 < nx//2 and abs(fcy) + spacings[i]/2 < ny//2:
            params.append([fcx, fcy, sr, sr, p])
            temp_ft = gabor_spectra(fx, fy, fcx, fcy, sr, sr, p)
            gabor_ft += np.abs(temp_ft[:, :])
        else:
            print("Skipped:", spacings[i], fcr[i])

params = np.array(params)
np.save("gabor_params_2d.npy", params)
plt.figure()
plt.imshow(gabor_ft)
#%%
# srs = 1/(2*np.clip(r, 1, None))
# spacings = 1.5/srs
def gabor_spectra(wx, wy, wt, fx, fy, ft, sx, sy, st, p):
    pt1 = np.exp(-1j*p-2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return pt1-pt2
def gabor_timedomain(x, y, t, fx, fy, ft, sx, sy, st, p):
    cx, cy, ct = 0.5, 0.5, 0.5
    exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = np.sin(2*np.pi*(x-cx)*fx+2*np.pi*(y-cy)*fy+2*np.pi*(t-ct)*ft+p)
    return exp*sin
n = 35
f = np.fft.fftshift(np.fft.fftfreq(n, 1/n))
print(f)
# f = np.linspace(-n//2, n//2 - int(n%2==0), n)
fx, fy, ft = np.meshgrid(f, f, f)
sp = np.linspace(0, 1, n)
x, y, t = np.meshgrid(sp, sp, sp)
r = np.sqrt(fx**2 + fy**2 + ft**2)
rmod = r[n//2, n//2:, n//2]
fcr, srs, spacings, nangles = [0], [1], [1.5], [1]
nextI = 0
while nextI < n//2:
    nextI = nextI + spacings[-1]
    srs.append(2/nextI)
    spacings.append(1.3/srs[-1])
    nangles.append(np.pi/np.arctan(spacings[-1]/2/nextI))
    fcr.append(nextI)
    
def hanning(r):
    return 0.5+0.5*np.cos(np.pi*r/(n//2))

gabor_ft = np.zeros((n, n))
fts = []
frmax = n/2
p = np.pi/2
picked_units = np.arange(len(fcr))#[0] + [i for i in range(1, len(fcr)) if spacings[i]/1.3 > 1]
radii = []
x_centers = []
y_centers = []
z_centers = []
sr_list = []
print("Picked units:", picked_units)
for i in picked_units:
    nspacings = 2*np.pi/nangles[i]
    phis = np.arange(nspacings/2*int(i%2==1), np.pi + nspacings/2*int(i%2==1), nspacings)
    for phiI, phi in enumerate(phis):
        thetas = np.arange(nspacings/2*int(i%2!=phiI%2), 2*np.pi + nspacings/2*int(i%2!=phiI%2), nspacings)
        for theta in thetas:
            fcx, fcy, fct = fcr[i]*np.cos(theta)*np.sin(phi), fcr[i]*np.sin(theta)*np.sin(phi), fcr[i]*np.cos(phi)
            sr = srs[i]
            sr_list.append(sr)
            radii.append(fcr[i])
            x_centers.append(fcx)
            y_centers.append(fcy)
            z_centers.append(fct)
            temp_ft = gabor_spectra(fx, fy, ft, fcx, fcy, fct, sr, sr, sr, p)
            gabor_ft += np.abs(temp_ft[:, :, n//2])
            fts.append(temp_ft)
            # gabor_ft = gabor_ft + gabor_spectra(fx, fy, ft, fcx, fcy, fct, sr, sr, sr, p)
# gabor = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(gabor_ft)))

for fi, ft in enumerate(fts):
    nrows = np.ceil(np.sqrt(n)).astype(int)
    ncols = np.ceil(n/nrows).astype(int)
    plt.figure(figsize=(ncols*2, nrows*2))
    # fted = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(ft))))
    fcx, fcy, fct = x_centers[fi], y_centers[fi], z_centers[fi]
    sr = sr_list[fi]
    fted = gabor_timedomain(x, y, t, fcx, fcy, fct, sr, sr, sr, p)
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(fted[:, :, i])
        plt.axis('off')
    plt.suptitle(f'rad{radii[fi]:.2f} | spacings = {spacings[fcr.index(radii[fi])]:.2f} |fcx, fcy, fct = {x_centers[fi]:.2f}, {y_centers[fi]:.2f}, {z_centers[fi]:.2f}', fontsize=8)
    plt.tight_layout()
    plt.gcf().set_facecolor('white')
    plt.savefig(f"fts/reals/ft_{fi}.png")
    plt.close('all')

plt.figure()
plt.imshow(gabor_ft)

#%%

# nrows = np.ceil(np.sqrt(len(radii))).astype(int)
ncols = np.ceil(len(radii)/nrows).astype(int)
plt.figure(figsize=(ncols*2, nrows*2))
for i in range(len(radii)):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fts[i])))))
    plt.title(f"({i}/{len(radii)})")
    plt.axis('off')
plt.tight_layout()

abs_ft = np.abs(gabor_ft)
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib widget
# Create figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.random.randn(len(radii))
colors = cm.jet(colors)
# Plot the spheres
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
outer_x = np.outer(np.cos(u), np.sin(v))
outer_y = np.outer(np.sin(u), np.sin(v))
outer_z = np.outer(np.ones(np.size(u)), np.cos(v))
for radius, x, y, z, c in zip(radii, x_centers, y_centers, z_centers, colors):
    x_surface = x + radius * outer_x
    y_surface = y + radius * outer_y
    z_surface = z + radius * outer_z
    ax.plot_surface(x_surface, y_surface, z_surface, color=c, alpha=0.2)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Overlapping Spheres')
ax.set_box_aspect((1, 1, 1))

# Show the plot
plt.show()
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
alphas = abs_ft[:, :, n//2:].ravel()
alphas = (alphas - 1).clip(0, None)/(alphas.max()-1)
ax.scatter(
    fx[:, :, n//2:].ravel(),
    fy[:, :, n//2:].ravel(),
    ft[:, :, n//2:].ravel(), alpha=alphas.tolist(), s=0.1)
# nrows = np.ceil(np.sqrt(n)).astype(int)
# ncols = np.ceil(n/nrows).astype(int)
# plt.figure(figsize=(ncols*2, nrows*2))
# for i in range(n):
#     plt.subplot(nrows, ncols, i+1)
#     plt.imshow(abs_ft[:, :, i], cmap='gray', vmin=abs_ft.min(), vmax=abs_ft.max())
#     plt.title(f"({i}/{n})")
#     plt.axis('off')
# plt.tight_layout() 
# plt.figure(figsize=(ncols*2, nrows*2))
# for i in range(n):
#     plt.subplot(nrows, ncols, i+1)
#     plt.imshow(abs_ft[:, :, i], cmap='gray')
#     plt.title(f"({i}/{n})")
#     plt.axis('off')
# plt.tight_layout() 
#%%

x = np.linspace(-0.5, 0.5, n)
x, y, t = np.meshgrid(x, x, x)

def fwhm(arr):
    # Get the full width half max of a curve
    return (arr>arr.max()/10).sum()
def gabor_spectra(wx, wy, wt, fx, fy, ft, sx, sy, st, p):
    pt1 = np.exp(-1j*p-2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return pt2#pt1-pt2
def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = np.sin(2*np.pi*(x-cx)*fx+2*np.pi*(y-cy)*fy+2*np.pi*(t-ct)*ft+p)
    return exp*sin
fc = 0
sx, sy, st = 15/n, 15/n, 15/n
p = 0

spacing = 3/2/sx
n_filters = n//spacing
f_tile_pos = np.linspace(0, n//2 - int(n%2==0), int(n_filters//2)+1) #np.arange(0, n//2 - int(n%2==0) + 1 - spacing/2, spacing)
f_tile_neg = np.linspace(-n//2, -spacing, int(n_filters//2))  #np.arange(-n//2+spacing/2, n//2 - int(n%2==0) + 1 - spacing/2, spacing) #np.arange(0, n//2 - int(n%2==0) - 1, 100/n)
f_tile = np.concatenate((f_tile_neg, f_tile_pos))
 #np.linspace(-n//2 + 2, n//2 - int(n%2==0) - 2, n//4)

nrows, ncols = len(f_tile), len(f_tile)
gabor_ft = 0
frmax = n/2
p = np.pi/2
i = 1
plt.figure(figsize=(ncols*2, nrows*2))
for fcx in f_tile:
    for fcy in f_tile:
        for fct in f_tile_pos[:1]:                
            sr = 0.5/np.clip(np.sqrt(fcx**2+fcy**2), 1, None)
            plt.subplot(nrows, ncols, i)
            i+=1
            if np.sqrt(fcx**2+fcy**2)+spacing/2 >= frmax:
                plt.title(f'bad-fx:{fcx:.1f};fy:{fcy:.1f};ft:{fct:.1f}')
                continue
            else:
                plt.title(f'fx:{fcx:.1f};fy:{fcy:.1f};ft:{fct:.1f}')
            
            plt.imshow(gabor_timedomain(x, y, t, fcx, fcy, fct, 0, 0, 0, sr, sr, sr, p)[..., n//2], vmin=-1, vmax=1)
            plt.axis('off')
            gabor_ft = gabor_ft + gabor_spectra(fx, fy, ft, fcx, fcy, fct, sr, sr, sr, p)[:, :, n//2]
plt.tight_layout()
#             gabor_ft = gabor_ft + gabor_spectra(fx, fy, ft, fcx, fcy, fc, sx, sy, st, p)[:, :, 0]
gabor = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(gabor_ft)))
plt.figure()
plt.title(fwhm(np.abs(gabor_ft[:, n//2])))
plt.imshow(np.abs(gabor_ft))
plt.figure()
plt.title(fwhm(np.abs(gabor[:, n//2])))
plt.imshow(np.abs(gabor))
#%%

n = 28
f = np.fft.fftshift(np.fft.fftfreq(n, 1/n))
print(f)
# f = np.linspace(-n//2, n//2 - int(n%2==0), n)
fx, fy, ft = np.meshgrid(f, f, f)

def fwhm(arr):
    # Get the full width half max of a curve
    return (arr>arr.max()/10).sum()
def gabor_spectra(wx, wy, wt, fx, fy, ft, sx, sy, st, p):
    pt1 = np.exp(-1j*p-2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return pt1-pt2
fc = 0
sx, sy, st = 15/n, 15/n, 15/n
p = 0

spacing = 3/2/sx
n_filters = n//spacing
f_tile_pos = np.linspace(0, n//2 - int(n%2==0), int(n_filters//2)+1) #np.arange(0, n//2 - int(n%2==0) + 1 - spacing/2, spacing)
f_tile_neg = np.linspace(-n//2, -spacing, int(n_filters//2))  #np.arange(-n//2+spacing/2, n//2 - int(n%2==0) + 1 - spacing/2, spacing) #np.arange(0, n//2 - int(n%2==0) - 1, 100/n)
f_tile = np.concatenate((f_tile_neg, f_tile_pos))
 #np.linspace(-n//2 + 2, n//2 - int(n%2==0) - 2, n//4)

gabor_ft = 0
for fcx in f_tile:
    for fcy in f_tile:
        for fct in f_tile_pos:
            gabor_ft = gabor_ft + gabor_spectra(fx, fy, ft, fcx, fcy, fc, sx, sy, st, p)[:, :, 0]
gabor = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(gabor_ft)))
plt.figure()
plt.title(fwhm(np.abs(gabor_ft[:, n//2])))
plt.imshow(np.abs(gabor_ft))
plt.figure()
plt.title(fwhm(np.abs(gabor[:, n//2])))
plt.imshow(np.abs(gabor))
#%%

def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = np.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)
    return exp*sin

def gabor_spectra(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    fx /= 2*np.pi
    fy /= 2*np.pi
    ft /= 2*np.pi
    constant = 1j*np.sqrt(2*np.pi**9)/(sx*sy*st)
    pt1 = np.exp(-1j*p-2*np.pi**2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*np.pi**2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return (pt1-pt2)*constant*np.exp(-1j*(wx*cx+wy*cy+wt*ct))

# Example time-domain signal
nx, ny, nt = 50, 50, 50
tstep = 4/nt
x = np.arange(-2, 2, tstep)
y = np.arange(-2, 2, tstep)
t = np.arange(-2, 2, tstep)
wx = np.fft.fftshift(np.fft.fftfreq(len(x), tstep))
wy = np.fft.fftshift(np.fft.fftfreq(len(y), tstep))
wt = np.fft.fftshift(np.fft.fftfreq(len(t), tstep))

fx, fy, ft = 3, 3, 3
x, y, t = np.meshgrid(x, y, t)
wx, wy, wt = np.meshgrid(wx, wy, wt)
cx, cy, ct = 0, 0, 0
sx, sy, st = 0.1, 0.1, 0.1
p = 0
signal = gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
signal_fft = np.fft.fftn(signal)
derived_ft = gabor_spectra(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
derived_signal = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(derived_ft)))

plt.figure(figsize=(10, 10))
# Plot the time-domain signal
plt.subplot(2, 2, 1)
plt.imshow(signal[:, :, nt//2])
plt.xlabel('Time')
plt.ylabel('Spatial Coordinate')
plt.title('Ground Truth Time-Domain Signal')

# Plot the ground truth Fourier transform
plt.subplot(2, 2, 2)
plt.imshow(np.abs(np.fft.fftshift(signal_fft))[:, :, nt//2])
plt.xlabel('Frequency')
plt.ylabel('Spatial Coordinate')
plt.title('Ground Truth Fourier Transform')

# Plot the time-domain transformed version of the derived Fourier transform
plt.subplot(2, 2, 3)
plt.imshow(np.real(derived_signal[:, :, nt//2]))
plt.xlabel('Time')
plt.ylabel('Spatial Coordinate')
plt.title('Derived Time-Domain Signal')

# Plot the derived Fourier transform
plt.subplot(2, 2, 4)
plt.imshow(np.abs(derived_ft[:, :, nt//2]))
plt.xlabel('Frequency')
plt.ylabel('Spatial Coordinate')
plt.title('Derived Fourier Transform')

# Show the plots
plt.tight_layout()
plt.show()

# %%
def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = torch.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = torch.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)
    return exp*sin
    
class Gabor(nn.Module):
    def __init__(self, input_dims, NC, max_batch_size=1000, frozen_center=False, bias=True):
        super().__init__()
        self.input_dims = input_dims[1:]
        self.NC = NC
        self.max_batch_size = max_batch_size
        self.gain = nn.Parameter(torch.ones(1, NC))
        self.bias = nn.Parameter(torch.zeros(1, NC)) if bias else 0
        #init parameters
        inits = torch.tensor([1, 1, 1, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0]).reshape(10, 1, 1, 1, 1)
        self.fparams = nn.Parameter(torch.ones(3, NC, 1, 1, 1)*inits[:3] + torch.randn(3, NC, 1, 1, 1)*0.01)
        self.spparams = nn.Parameter(torch.ones(4, NC, 1, 1, 1)*inits[6:10] + torch.randn(4, NC, 1, 1, 1)*0.01)
        cparams = torch.ones(3, NC, 1, 1, 1)*inits[3:6]+torch.randn(3, NC, 1, 1, 1)*0.01*int(not frozen_center)
        self.cparams = nn.Parameter(cparams, requires_grad=not frozen_center)
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, *self.input_dims)
        gabors = self.get_kernels()
        if len(x) > self.max_batch_size:
            return torch.cat([torch.sum(x[i:i+self.max_batch_size]*gabors, dim=(2, 3, 4)) for i in range(0, len(x), self.max_batch_size)])
        return torch.sum(x*gabors, dim=(2, 3, 4)) * self.gain + self.bias
    def get_kernels(self):
        gridx = torch.linspace(0, 1, self.input_dims[0], device=self.fparams.device).reshape(1, -1, 1, 1)
        gridy = torch.linspace(0, 1, self.input_dims[1], device=self.fparams.device).reshape(1, 1, -1, 1)
        gridt = torch.linspace(0, 1, self.input_dims[2], device=self.fparams.device).reshape(1, 1, 1, -1)
        return gabor_timedomain(gridx, gridy, gridt, *self.fparams, *self.cparams, *self.spparams)

class GaborGLM(nn.Module):
    def __init__(self, input_dims, NC, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Sequential(Gabor(input_dims, NC))
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.core(x))
    def get_filters(self):
        return self.core[0].get_kernels().detach().cpu()
    
class GaborBasisGLM(nn.Module):
    def __init__(self, input_dims, basis_dim, NC, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Sequential(GaborGLM(input_dims, basis_dim, output_NL=nn.Identity()))
        self.readout = nn.Linear(basis_dim, NC)
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.readout(self.core(x)))
    
class GaborConvLayer(nn.Module):
    # expects to be shaped (batch, cin, x, y, t)
    def __init__(self, cin, cout, k=3, **kwargs):
        super().__init__()
        k = (k, k, k) if isinstance(k, int) else k
        self.gabor = Gabor((1, *k), cin*cout, frozen_center=True)
        self.cout = cout
        self.cin = cin
        self.kwargs = kwargs
    def forward(self, x):
        kernels = self.gabor.get_kernels()
        return F.conv3d(x, kernels.reshape(self.cout, self.cin, *kernels.shape[1:]), **self.kwargs)
    
class GaborConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, NL=nn.Softplus(), **kwargs):
        super().__init__()
        self.conv = GaborConvLayer(cin, cout, k=k, **kwargs)
        self.NL = NL
        self.bn = nn.BatchNorm3d(cin)
    def forward(self, x):
        return self.NL(self.conv(self.bn(x)))

class FoldTimetoChannel(nn.Module):
    def forward(self, x):
        return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], x.shape[1]*x.shape[4], x.shape[2], x.shape[3])
class Reshape(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims
    def forward(self, x):
        return x.reshape(x.shape[0], *self.input_dims)
class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x