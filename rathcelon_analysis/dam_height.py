import pandas as pd
import matplotlib.pyplot as plt

lhd_df = pd.read_csv("C:/Users/ki87ujmn/Downloads/LHD_RathCelon/LowHead_Dam_Database.csv")

fig, ax = plt.subplots()

cross_sections = {
    "P_1": ("lightblue", "o"),
    "P_2": ("green", "s"),
    "P_3": ("cyan", "^"),
    "P_4": ("dodgerblue", "d"),
}

counter = 0
for XS, (color, marker) in cross_sections.items():
    counter += 1
    ax.scatter(lhd_df["P_known"], lhd_df[XS],
               color=color, marker=marker, label=f'Cross-Section {counter}', alpha=0.8)

# Plot the 1:1 line (perfect agreement)
min_val = 0
max_val = max(lhd_df['P_known'])
ax.plot([min_val, max_val], [min_val, max_val], linestyle='-', color='black', label='Perfect Agreement')

# Labels and legend
ax.set_xlabel("NID Heights (ft)")
ax.set_ylabel("Estimated Values (ft)")
ax.set_title("Estimated Dam Height Against National Inventory of Dams")
ax.legend()
ax.grid(True)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
