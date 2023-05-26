#%%
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline


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

fx, fy, ft = 5.5, 5.5, 5.5
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
tlim = 1 #seconds
n = 50
n_show = 30
fs = n / tlim #Hz
flim = fs / 2
print(flim, tlim)
fx, fy, ft = flim/4, flim/4, flim/4
cx, cy, ct = 0, 0, 0
sx, sy, st = 0.5, 0.5, 0.5
p = 0
time_domain = np.linspace(-tlim, tlim, n)
freqs = np.linspace(-flim, flim, n)
wx, wy, wt = np.meshgrid(freqs, freqs, freqs)
x, y, t = np.meshgrid(time_domain, time_domain, time_domain)
gabor_ft = sin_ft(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
real_gabor = np.real(np.fft.fftshift(np.fft.ifftn(gabor_ft, (n*4, n*4, n), norm='backward')))
gt = sin_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
gt_ft = np.abs(np.fft.fftshift(np.fft.fftn(gt, norm='forward')))
#plot the real_gabor filter
numrows = math.ceil(math.sqrt(real_gabor.shape[-1]))
numcols = math.ceil(real_gabor.shape[-1] / numrows)
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    # plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    plt.imshow(np.abs(gt_ft[..., i]), cmap='gray')
    # plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    # plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    # plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    plt.imshow(gt[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
# %%

#%%
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline

def is_val(x, val):
    # x is nd array, return the value in x closest to val
    return x == x.flat[np.argmin(np.abs(x-val))]

def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = np.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)
    return exp*sin

def sin_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    return np.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)

def sin_ft(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    fx *= 2*np.pi
    fy *= 2*np.pi
    ft *= 2*np.pi
    outgrid = np.zeros(wx.shape, dtype=np.complex128)
    phase = 4*np.pi**3*1j*np.exp(1j*(p-fx*cx-fy*cy-ft*ct))
    phase_neg = -4*np.pi**3*1j*np.exp(-1j*(p-fx*cx-fy*cy-ft*ct))
    # inds_pos = is_val(wx, fx) & is_val(wy, fy) & is_val(wt, ft)
    # inds_neg = is_val(wx, -fx) & is_val(wy, -fy) & is_val(wt, -ft)
    # print(inds_pos.sum(), inds_neg.sum())
    # print(wx[inds_pos], wy[inds_pos], wt[inds_pos])
    # print(wx[inds_neg], wy[inds_neg], wt[inds_neg])
    outgrid[is_val(wx, fx) & is_val(wy, fy) & is_val(wt, ft)] = phase
    outgrid[is_val(wx, -fx) & is_val(wy, -fy) & is_val(wt, -ft)] = phase_neg
    return outgrid

def gabor_spectra(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    #wx.. are grids that tell us the freq at each point
    # outgrid = np.zeros(wx.shape, dtype=np.complex128)
    fx /= 2*np.pi
    fy /= 2*np.pi
    ft /= 2*np.pi
    constant = 1j*np.sqrt(2*np.pi**9)/(sx*sy*st)
    pt1 = np.exp(-1j*p-2*np.pi**2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    pt2 = np.exp(1j*p-2*np.pi**2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    return (pt1-pt2)*constant*np.exp(-1j*(wx*cx+wy*cy+wt*ct))
    
    # constant = 1j*np.sqrt(2*np.pi**9)/(sx*sy*st)
    # pt1 = np.exp(-1j*p-2*np.pi**2*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    # pt2 = np.exp(1j*p-2*np.pi**2*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    # return constant*np.exp(-1j*(wx*cx+wy*cy+wt*ct))*(pt1-pt2)

    # constant = 1j*8*np.sqrt(2*np.pi**9)*(sx*sy*st)
    # pt1 = np.exp(-1j*p-0.5*(sx**2*(wx+fx)**2+sy**2*(wy+fy)**2+st**2*(wt+ft)**2))
    # pt2 = np.exp(1j*p-0.5*(sx**2*(wx-fx)**2+sy**2*(wy-fy)**2+st**2*(wt-ft)**2))
    # return constant*np.exp(-1j*(wx*cx+wy*cy+wt*ct))*(pt1-pt2)
    
    # constant = 4j*np.pi**3.5*math.sqrt(2)*sx*sy*st*np.exp(-1j*(cx*sx+cy*sy+ct*st))
    # phaser = np.exp(1j*(p-cx*fx-cy*fy-ct*ft))
    # phaser_neg = np.exp(-1j*(p-cx*fx-cy*fy-ct*ft))
    # pt1 = np.exp(-0.5*sx**2*(wx+fx)**2-0.5*sy**2*(wy+fy)**2-0.5*st**2*(wt+ft)**2)
    # pt2 = np.exp(-0.5*sx**2*(wx-fx)**2-0.5*sy**2*(wy-fy)**2-0.5*st**2*(wt-ft)**2)
    # return constant*(phaser_neg*pt1 - phaser*pt2)
    # fx_pos, fy_pos, ft_pos = get_closest(wx, fx), get_closest(wy, fx), get_closest(wt, fx)
    # fx_neg, fy_neg, ft_neg = get_closest(wx, -fx), get_closest(wy, -fx), get_closest(wt, -fx)
    # print(fx_pos, fy_pos, ft_pos, fx_neg, fy_neg, ft_neg)
    # print(((wt==ft_neg) & (wy==fy_neg)).sum(), ((wt==ft_neg) & (wx==fx_neg)).sum(), ((wy==fy_neg) & (wx==fx_neg)).sum())
    # print(((wt==ft_pos) & (wy==fy_pos)).sum(), ((wt==ft_pos) & (wx==fx_pos)).sum(), ((wy==fy_pos) & (wx==fx_pos)).sum())
    # outgrid += constant*phaser_neg*sx*np.exp(-1j*cx*sx)*np.exp(-0.5*sx**2*(wx+fx)**2)
    # outgrid += constant*phaser_neg*sy*np.exp(-1j*cy*sy)*np.exp(-0.5*sy**2*(wy+fy)**2)
    # outgrid += constant*phaser_neg*st*np.exp(-1j*ct*st)*np.exp(-0.5*st**2*(wt+ft)**2)
    # outgrid += constant*phaser*sx*np.exp(-1j*cx*sx)*np.exp(-0.5*sx**2*(wx-fx)**2)
    # outgrid += constant*phaser*sy*np.exp(-1j*cy*sy)*np.exp(-0.5*sy**2*(wy-fy)**2)
    # outgrid += constant*phaser*st*np.exp(-1j*ct*st)*np.exp(-0.5*st**2*(wt-ft)**2)
    # outgrid[(wt==ft_neg) & (wy==fy_neg)] += constant*phaser_neg*sx*np.exp(-1j*cx*sx)*np.exp(-0.5*sx**2*(wx[(wt==ft_neg) & (wy==fy_neg)]+fx)**2)
    # outgrid[(wt==ft_neg) & (wx==fx_neg)] += constant*phaser_neg*sy*np.exp(-1j*cy*sy)*np.exp(-0.5*sy**2*(wy[(wt==ft_neg) & (wx==fx_neg)]+fy)**2)
    # outgrid[(wy==fy_neg) & (wx==fx_neg)] += constant*phaser_neg*st*np.exp(-1j*ct*st)*np.exp(-0.5*st**2*(wt[(wy==fy_neg) & (wx==fx_neg)]+ft)**2)
    # outgrid[(wt==ft_pos) & (wy==fy_pos)] += constant*phaser*sx*np.exp(-1j*cx*sx)*np.exp(-0.5*sx**2*(wx[(wt==ft_pos) & (wy==fy_pos)]-fx)**2)
    # outgrid[(wt==ft_pos) & (wx==fx_pos)] += constant*phaser*sy*np.exp(-1j*cy*sy)*np.exp(-0.5*sy**2*(wy[(wt==ft_pos) & (wx==fx_pos)]-fy)**2)
    # outgrid[(wy==fy_pos) & (wx==fx_pos)] += constant*phaser*st*np.exp(-1j*ct*st)*np.exp(-0.5*st**2*(wt[(wy==fy_pos) & (wx==fx_pos)]-ft)**2)
    
    return outgrid

# Example time-domain signal
nx, ny, nt = 50, 50, 50
tstep = 4/nt
x = np.arange(-2, 2, tstep)
y = np.arange(-2, 2, tstep)
t = np.arange(-2, 2, tstep)
wx = np.fft.fftshift(np.fft.fftfreq(len(x), tstep))
wy = np.fft.fftshift(np.fft.fftfreq(len(y), tstep))
wt = np.fft.fftshift(np.fft.fftfreq(len(t), tstep))

fx, fy, ft = 5.5, 5.5, 5.5
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
tlim = 1 #seconds
n = 50
n_show = 30
fs = n / tlim #Hz
flim = fs / 2
print(flim, tlim)
fx, fy, ft = flim/4, flim/4, flim/4
cx, cy, ct = 0, 0, 0
sx, sy, st = 0.5, 0.5, 0.5
p = 0
time_domain = np.linspace(-tlim, tlim, n)
freqs = np.linspace(-flim, flim, n)
wx, wy, wt = np.meshgrid(freqs, freqs, freqs)
x, y, t = np.meshgrid(time_domain, time_domain, time_domain)
gabor_ft = sin_ft(wx, wy, wt, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
real_gabor = np.real(np.fft.fftshift(np.fft.ifftn(gabor_ft, (n*4, n*4, n), norm='backward')))
gt = sin_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p)
gt_ft = np.abs(np.fft.fftshift(np.fft.fftn(gt, norm='forward')))
#plot the real_gabor filter
numrows = math.ceil(math.sqrt(real_gabor.shape[-1]))
numcols = math.ceil(real_gabor.shape[-1] / numrows)
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    # plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    plt.imshow(np.abs(gt_ft[..., i]), cmap='gray')
    # plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    # plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    plt.imshow(real_gabor[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.figure(figsize=(numcols*2, numrows*2))
for i in range(n_show):
    plt.subplot(numrows, numcols, i+1)
    # plt.imshow(np.abs(gabor_ft[..., i]), cmap='gray')
    plt.imshow(gt[..., i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
# %%
