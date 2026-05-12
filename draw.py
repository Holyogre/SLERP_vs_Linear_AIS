import pandas as pd
import matplotlib.pyplot as plt

LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

df = pd.read_csv("interp_k_summary.csv")

df_linear = df[df["Method"] == "linear"].sort_values("k")
df_slerp  = df[df["Method"] == "slerp"].sort_values("k")

k_values = df_linear["k"].values
mae_linear = df_linear["MAE_m"].values
mae_slerp  = df_slerp["MAE_m"].values
mae_min = min(mae_linear.min(), mae_slerp.min())

plt.figure(figsize=(8, 5))

plt.plot(
    k_values, mae_linear, 
    marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=6,
    label="线性插值"
)
plt.plot(
    k_values, mae_slerp, 
    marker='s', linestyle='-', color='#ff4545', linewidth=2, markersize=6,
    label="SLERP插值"
)

plt.xticks(k_values, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlabel("插值间隔控制参数 $k$", fontsize=LABEL_SIZE)
plt.ylabel("平均绝对误差（米）", fontsize=LABEL_SIZE)
plt.xlim(left=k_values.min())
plt.ylim(bottom=mae_min)


plt.legend(fontsize=LEGEND_SIZE)
plt.grid(alpha=0.2, linestyle='--')
plt.tight_layout()

plt.savefig("mae_compare.png", dpi=300)
plt.show()

improvement = (mae_linear - mae_slerp) 

plt.figure(figsize=(8, 5))
plt.plot(k_values, improvement, marker='s', linewidth=3, color='#2ca02c', label="MAE差值")
plt.axhline(0, color='gray', linestyle='--')

plt.xticks(k_values, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlabel("插值间隔控制参数$k$", fontsize=LABEL_SIZE)
plt.ylabel("MAE差值（米）", fontsize=LABEL_SIZE)
plt.xlim(left=k_values.min(), right=k_values.max()+1) 
plt.ylim(bottom=0)  
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("mae_improvement.png", dpi=300)
plt.show()