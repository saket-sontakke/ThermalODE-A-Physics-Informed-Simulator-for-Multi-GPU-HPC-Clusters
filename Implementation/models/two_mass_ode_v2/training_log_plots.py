import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def generate_plot_grid(csv_filename='training_log.csv', output_filename='parameter_plots_grid.svg'):
    # Check if file exists
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}' in the current directory.")
        return

    # Load the CSV data
    df = pd.read_csv(csv_filename)
    
    # 1. Restrict to epoch 81 and earlier
    df = df[df['epoch'] <= 81]
    epochs = df['epoch']

    # 2. Rename columns from old CSV format to the exact required parameter names
    # (Matches the names in self.bounds)
    rename_dict = {
        'C_die0': 'C_die_0',       'C_die1': 'C_die_1',
        'C_sink0': 'C_sink_0',     'C_sink1': 'C_sink_1',
        'R_pst0': 'R_paste_0',     'R_pst1': 'R_paste_1',
        'h_base0': 'h_base_0',     'h_base1': 'h_base_1',
        'h_act0': 'h_active_0',    'h_act1': 'h_active_1',
        'T_thr0': 'T_thresh_0',    'T_thr1': 'T_thresh_1',
        'beta0': 'beta_0',         'beta1': 'beta_1'
    }
    df.rename(columns=rename_dict, inplace=True)

    # Define the color scheme from your SVG
    COLOR_0 = '#07B486'  # Teal/Green for Train or Parameter 0
    COLOR_1 = '#F39C12'  # Orange for Val or Parameter 1
    TEXT_COLOR_TITLE = '#2F3D4A'
    TEXT_COLOR_AXIS = '#6F7E8C'
    GRID_COLOR = '#F0F0F0'

    # Define groupings using the newly mapped names
    # Added 'beta' group just in case it exists in your logs based on your bounds
    groups = [
        ('Loss Curve (RMSE)', ['train_rmse', 'val_rmse']),
        ('Loss Curve (MAE)', ['train_mae', 'val_mae']),
        ('Learning Rate', ['lr']),
        ('C_die', ['C_die_0', 'C_die_1']),
        ('C_sink', ['C_sink_0', 'C_sink_1']),
        ('R_paste', ['R_paste_0', 'R_paste_1']),
        ('Cross-Coupling (k)', ['k01', 'k10']),
        ('h_base', ['h_base_0', 'h_base_1']),
        ('h_active', ['h_active_0', 'h_active_1']),
        ('T_thresh', ['T_thresh_0', 'T_thresh_1']),
        ('q', ['q0', 'q1']),
        ('beta', ['beta_0', 'beta_1']) 
    ]

    # Filter out groups where NONE of the columns exist in the dataframe
    # (e.g., if 'beta' isn't actually in the CSV, it skips the plot)
    groups = [(title, cols) for title, cols in groups if any(c in df.columns for c in cols)]

    # Calculate grid dimensions 
    n_plots = len(groups)
    cols = 3
    rows = math.ceil(n_plots / cols)

    # Initialize the figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() # Flatten to iterate easily

    for i, (title, columns) in enumerate(groups):
        ax = axes[i]
        
        # Plot first parameter (0 / Train)
        if columns[0] in df.columns:
            ax.plot(epochs, df[columns[0]], label=columns[0], color=COLOR_0, linewidth=2)
        
        # Plot second parameter (1 / Val) if it exists in the grouping
        if len(columns) > 1 and columns[1] in df.columns:
            ax.plot(epochs, df[columns[1]], label=columns[1], color=COLOR_1, linewidth=2)

        # Apply SVG-matching Aesthetics
        ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_COLOR_TITLE, pad=15)
        ax.set_xlabel('Epoch', fontsize=12, color=TEXT_COLOR_AXIS)
        ax.set_ylabel('Value', fontsize=12, color=TEXT_COLOR_AXIS)
        
        # Adjust tick parameters
        ax.tick_params(axis='x', colors=TEXT_COLOR_TITLE)
        ax.tick_params(axis='y', colors=TEXT_COLOR_TITLE)

        # Clean up the spines (borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E0E0E0')
        
        # Horizontal gridlines only
        ax.yaxis.grid(True, color=GRID_COLOR, linestyle='-', linewidth=1)
        ax.set_axisbelow(True) # Ensure grid is behind the lines

        # 3. Use loc='best' to automatically avoid overlapping the legend with the graph lines
        ax.legend(frameon=False, labelcolor=TEXT_COLOR_AXIS, loc='best')

    # Remove any empty subplots if the grid isn't perfectly filled
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent clipping and save
    plt.tight_layout(pad=3.0)
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"Success! Grid saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_plot_grid()