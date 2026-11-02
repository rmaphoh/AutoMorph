# About this fork / このフォークについて

**日本語:** このリポジトリは [rmaphoh/AutoMorph](https://github.com/rmaphoh/AutoMorph) の非公式のフォークです。元のコードは変更せず、M3（形態指標の計算）を原著の定義に合うように書き直した版を [`AutoMorph_fork/`](AutoMorph_fork/) に追加しました。主な改正点は次のとおりです。各改正の根拠、引用、変更していない点、限界は [AutoMorph_fork/README.md](AutoMorph_fork/README.md) にあります。

**English:** This repository is an unofficial fork of [rmaphoh/AutoMorph](https://github.com/rmaphoh/AutoMorph). The original code is unchanged, and a version of M3 (morphology measurement) rewritten to follow the definitions in the original papers is added in [`AutoMorph_fork/`](AutoMorph_fork/). The main changes are listed below. The evidence and citations for each change, what was not changed, and the limitations are in [AutoMorph_fork/README.md](AutoMorph_fork/README.md).

| 対象 / Item | 元のコード / Original code | この版 / This version | 根拠 / Evidence |
|:---|:---|:---|:---|
| 点の順序 / Point order | 幅優先探索の訪問順 / Breadth-first visiting order | 血管の端から順に / From one vessel end to the other | Hart 1999; Grisan 2008 |
| 曲線 / Curve | 画素の点の列 / Pixel chain | 平滑化スプライン / Smoothing spline | Grisan 2003, 2008 |
| 曲率の蛇行度 / Curvature tortuosity | \|∫κ\| | ∫κ²ds | Hart 1999 p.244; Grisan 2003 eq. 15 |
| 蛇行密度 / Tortuosity density | `(n-1)/n + (1/L)Σ(…)` | `(n-1)/n × (1/L)Σ(…)` | Grisan 2003 eq. 14; Grisan 2008 eq. 19 |
| 変曲点 / Inflection points | y の増減の変化 / Changes in the direction of y | 曲率の符号の変化 / Sign changes of curvature | Grisan 2003, 2008 |
| 血管幅 / Vessel width | 面積÷骨格の画素数 / Area ÷ skeleton pixel count | 面積÷中心線の長さ / Area ÷ centerline length | W/cosθ の偏り / W/cosθ bias on diagonals |
| CRAE/CRVE の幅 / width | 円環で切ったマスク / Mask cut by the annulus | 切る前のマスク / Uncut mask | 縁で幅が小さく出る / Width is lowered at the edge |

M0–M2, FD, vessel density, and the Knudtson formula are unchanged. / M0〜M2、FD、血管密度、Knudtson の式は変更していません。

The original AutoMorph README follows. / 以下は元の AutoMorph の README です。

---

# AutoMorph 2022 👀
--Code for [AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline](https://tvst.arvojournals.org/article.aspx?articleid=2783477).

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.

Project website: https://rmaphoh.github.io/projects/automorph.html

Talks on NIHR Moorfields BRC: https://moorfieldsbrc.nihr.ac.uk/case-study/research-report/



## News 👀
2025-05-16 update: Docker and its instruction have been updated.

2024-06-27 update: pytorch 2.3 & python 3.11 supported; Mac M2 GPU supported; CPU supported (thanks to [staskh](https://github.com/staskh))

2023-08-24 update: Added feature measurement for disc-centred images; removed unused files.
&nbsp;




## Pixel resolution

The units for vessel average width, disc/cup height and width, and calibre metrics are defined as microns. For it, we need to organise a [resolution_information.csv](https://github.com/rmaphoh/AutoMorph/blob/main/resolution_information.csv) which includes the pixel resolution information, which can be queried in FDA or Dicom files. Alternatively, approximate value can be used, e.g., 0.008 for Topcon 3D-OCT.

**If you don't use these features or care their units**, you can put all images in the folder ./images and run

```bash
python generate_resolution.py
```
&nbsp;


## Running AutoMorph
### Running with Colab

Use the Google Colab and a free Tesla T4 gpu [Colab link click](https://colab.research.google.com/drive/13Qh9umwRM1OMRiNLyILbpq3k9h55FjNZ?usp=sharing).

👀 A specific version for [APTOSxJSAIO 2025 hands-on tutorial](https://colab.research.google.com/drive/1ppzxBElLa_yJl-V3IWKyQQDG_2m6OWVL#scrollTo=vnEudwJimFgt)


### Running on local/virtual machine

Install and use on your own machines [LOCAL.md](LOCAL.md).


### Running with Docker

Zero experience in Docker? No worries [DOCKER.md](DOCKER.md).

&nbsp;

## Common questions

### Memory/ram error

We use Tesla T4 (16Gb) and 32vCPUs (120Gb). When you meet memory/ram issue in running, try to decrease batch size:

* ./M1_Retinal_Image_quality_EyePACS/test_outside.sh -b=64 to smaller, e.g., 32 or 16.
* ./M2_Artery_vein/test_outside.sh --batch-size=8 to smaller
* ./M2_lwnet_disc_cup/test_outside.sh --batchsize=8 to smaller


### Invalid results

In csv files, invalid values (e.g., optic disc segmentation failure) are indicated with NAN values.  


### Components

1. Vessel segmentation [BF-Net](https://github.com/rmaphoh/Learning-AVSegmentation.git)

2. Image pre-processing [EyeQ](https://github.com/HzFu/EyeQ.git) 

3. Optic disc segmentation [lwnet](https://github.com/agaldran/lwnet.git)

4. Feature measurement [retipy](https://github.com/alevalv/retipy.git)

&nbsp;

## Citation

```
@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

