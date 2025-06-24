import matplotlib.pyplot as plt

C_baseline_BLOOM = 471000000
C_1_BLOOM = 2140000000
C_2_BLOOM = 2140000000
C_3_BLOOM = 2140000000
CpromptBLOOM_baseline = 1.57
CpromptBLOOM1 = 7.14
CpromptBLOOM2 = 7.14
CpromptBLOOM3 = 7.14
CpageBLOOM_baseline = 0.95
CpageBLOOM1 = 7.14
CpageBLOOM2 = 21.4
CpageBLOOM3 = 21.4

C_baseline_CGPT = 6.66e8
C_1_CGPT = 6.29e9
C_2_CGPT = 6.29e9
C_3_CGPT = 6.29e9
C_4_CGPT = 2.16e12
C_5_CGPT = 2.16e12
CpromptCGPT_baseline = 2.22
CpromptCGPT1 = 20.96
CpromptCGPT2 = 20.96
CpromptCGPT3 = 20.96
CpromptCGPT4 = 591.04
CpromptCGPT5 = 591.04
CpageCGPT_baseline = 1.34
CpageCGPT1 = 20.96
CpageCGPT2 = 62.9
CpageCGPT3 = 62.9
CpageCGPT4 = 591.04
CpageCGPT5 = 1773.31


human_baseline = 1426
human1 = 22.63
human2 = 22.63
human3 = 4.92
human4 = 1426
human5 = 1426

ratioCpageBLOOM_baseline: 1501
ratioCpageBLOOM1 = 3.16
ratioCpageBLOOM2 = 1.05
ratioCpageBLOOM3 = 0.23

ratioCpageCGPT_baseline: 1501
ratioCpageCGPT1 = 3.16
ratioCpageCGPT2 = 1.05
ratioCpageCGPT3 = 0.23
ratioCpageCGPT4 = 0.23
ratioCpageCGPT5 = 0.23

x_title = "Scenarios"
x_labels = ["B", "S1", "S2", "S3", "S4", "S5"]

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8,10))
ax[0][0].bar([0,1,2], [471000000, 2140000000, 2140000000], facecolor='0.7')
ax[0][0].set_xlabel("Scenario")
ax[0][0].set_xticks([0,1,2])
ax[0][0].set_xticklabels(x_labels[0:3])
ax[0][0].set_ylabel("C (g CO2e)")
ax[0][0].set_title("BLOOM carbon footprint")

ax[0][1].bar([0,1,2], [0.95, 7.14, 21.4], facecolor='0.7')
ax[0][1].set_xlabel("Scenario")
ax[0][1].set_xticks([0,1,2])
ax[0][1].set_xticklabels(x_labels[0:3])
ax[0][1].set_ylabel("Cp (gCO2e / page)")
ax[0][1].set_title("BLOOM carbon per page")

ax[1][0].bar([0,1,2,3,4], [6.66e8, 6.29e9, 6.29e9, 2.16e10, 2.16e10], facecolor='0.7')
ax[1][0].set_xlabel("Scenario")
ax[1][0].set_xticks([0,1,2,3,4])
ax[1][0].set_xticklabels(x_labels[0:5])
ax[1][0].set_ylabel("C (g CO2e)")
ax[1][0].set_title("Chat-GPT carbon footprint")
ax[1][0].text(3.75, 1.8e10,"x100")
ax[1][0].text(2.75, 1.8e10,"x100")

ax[1][1].bar([0,1,2,3,4], [1.34, 20.96, 62.9, 59.1, 177.3], facecolor='0.7')
ax[1][1].set_xlabel("Scenario")
ax[1][1].set_xticks([0,1,2,3,4])
ax[1][1].set_xticklabels(x_labels[0:5])
ax[1][1].set_ylabel("Cp (gCO2e / page)")
ax[1][1].set_title("Chat-GPT carbon per page")
ax[1][1].text(3.75, 35,"x10")
ax[1][1].text(2.75, 35,"x10")

ax[2][0].bar([0,1,2], [15.01, 3.16, 1.05], facecolor='0.7')
ax[2][0].set_xlabel("Scenario")
ax[2][0].set_xticks([0,1,2,3])
ax[2][0].set_xticklabels(x_labels[0:3])
ax[2][0].set_ylabel("ratio")
ax[2][0].set_title("Ratio Cp human:AI, BLOOM")
ax[2][0].text(-0.2,12,"x100")

ax[2][1].bar([0,1,2], [10.64, 1.07, 0.36], facecolor='0.7')
ax[2][1].set_xlabel("Scenario")
ax[2][1].set_xticks([0,1,2,3])
ax[2][1].set_xticklabels(x_labels[0:3])
ax[2][1].set_ylabel("ratio")
ax[2][1].set_title("Ratio Cp human:AI, Chat-GPT")

ax[2][1].text(-0.2,9,"x100")


fig.tight_layout()
plt.savefig("fig1.png")
plt.show()
