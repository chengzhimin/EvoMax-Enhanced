# ESM-IF1 权重上传位置

请将官方 ESM-IF1 checkpoint 上传为：

```text
/data/run01/scwb286/esm2_deploy/weights/ESM-IF1/esm_if1_gvp4_t16_142M_UR50.pt
```

同时准备 WT 结构文件，例如：

```text
/data/run01/scwb286/EvoMax-Enhanced/data/raw/wildtype.pdb
```

结构链必须与 WT FASTA 一一对应。上传后不要改名；程序会用文件存在性、链名和结构序列一致性检查。
