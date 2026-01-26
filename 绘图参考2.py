import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

projn = ccrs.LambertConformal(
    central_longitude=105,
    central_latitude=40,
    standard_parallels=(25.0, 47.0),
)

fig = plt.figure(figsize=(6, 7), dpi=200, facecolor="w")
ax = fig.add_subplot(projection=projn)
ax.add_feature(cfeature.LAND, facecolor="white")
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="#BEE8FF")
ax.set_extent([70, 140, 15, 52], crs=ccrs.PlateCarree())

long = np.linspace(72, 136, 128)
lat = np.linspace(18, 54, 72)

# 创建颜色映射器
cmap = plt.get_cmap("RdYlBu_r")
norm = mcolors.Normalize(vmin=0, vmax=2000)

im = ax.contourf(
    long,
    lat[::-1],
    prcp1,
    levels=np.linspace(0, 2000, 11),
    cmap=cmap,
    extend="both",
    alpha=0.8,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    im,
    ax=ax,
    shrink=0.9,
    pad=0.1,
    orientation="horizontal",
)

add_dashline(ax, ec="black", linewidth=0.2)
add_china(ax, ec="black", fc="None", linewidth=0.2)

gls = ax.gridlines(
    draw_labels=True,
    crs=ccrs.PlateCarree(),
    color="k",
    linestyle="dashed",
    linewidth=0.3,
    y_inline=False,
    x_inline=False,
    xlocs=range(70, 150, 10),
    ylocs=range(15, 65, 10),
)

# 创建南海诸岛小图
ax2 = fig.add_axes([0.7, 0.258, 0.2, 0.3], projection=projn)  # left, bottom, width, height
ax2.set_extent([104.5, 125, 0, 26])
im2 = ax2.contourf(
    long,
    lat[::-1],
    prcp1,
    levels=np.linspace(0, 2000, 11),
    cmap=cmap,
    extend="both",
    alpha=0.8,
    transform=ccrs.PlateCarree(),
)
ax2.set_facecolor("#BEE8FF")
ax2.spines["geo"].set_linewidth(0.2)
lb = ax2.gridlines(
    draw_labels=False,
    x_inline=False,
    y_inline=False,
    linewidth=0.1,
    color="gray",
    alpha=0.8,
    linestyle="--",
)
add_dashline(ax2, ec="black", linewidth=0.2)
add_china(ax2, ec="black", fc="None", linewidth=0.2)
ax2.add_feature(cfeature.LAND, facecolor="w")
ax2.add_feature(cfeature.OCEAN)
ax2.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="#BEE8FF")

# 计算各个区间的像素值数量
levels = np.linspace(0, 2000, 11)
hist, bins = np.histogram(prcp1, bins=levels)

# 在左下角添加统计图
ax_hist = fig.add_axes([0.17, 0.33, 0.18, 0.09])  # left, bottom, width, height
for i in range(len(hist)):
    ax_hist.bar(
        bins[i],
        hist[i],
        width=np.diff(bins)[i],
        color=cmap(norm(bins[i])),
        edgecolor="black",
        alpha=0.7,
    )

ax_hist.set_xlim(-100, 2000)  # 稍微扩大左边界
ax_hist.tick_params(axis="both", which="both", length=0)
ax_hist.set_xticklabels([])
ax_hist.set_yticklabels([])
ax_hist.grid(False)

plt.show()
