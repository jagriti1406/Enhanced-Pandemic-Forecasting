import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


gnn_dir = r"C:\Users\amanl\Downloads\pandemic_tgnn-master\pandemic_tgnn-master\code\output_GNN"
hybrid_dir = r"C:\Users\amanl\Downloads\pandemic_tgnn-master\pandemic_tgnn-master\code\output_Hybrid"

def calculate_metrics(directory, model_name):
    metrics = {'MAE': [], 'MSE': [], 'R2': [], 'Model': []}
    total_mae, total_mse, total_r2 = 0, 0, 0
    file_count = 0

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_count += 1
            df = pd.read_csv(os.path.join(directory, file))
            
            mae = abs(df['o'] - df['l']).mean()
            mse = ((df['o'] - df['l']) ** 2).mean()
            ss_total = ((df['l'] - df['l'].mean()) ** 2).sum()
            ss_residual = ((df['l'] - df['o']) ** 2).sum()
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

            total_mae += mae
            total_mse += mse
            total_r2 += r2

            metrics['MAE'].append(mae)
            metrics['MSE'].append(mse)
            metrics['R2'].append(r2)
            metrics['Model'].append(model_name)

    avg_mae = total_mae / file_count if file_count > 0 else 0
    avg_mse = total_mse / file_count if file_count > 0 else 0
    avg_r2 = total_r2 / file_count if file_count > 0 else 0

    return avg_mae, avg_mse, avg_r2, file_count, pd.DataFrame(metrics)


gnn_mae, gnn_mse, gnn_r2, gnn_files, gnn_metrics_df = calculate_metrics(gnn_dir, 'GNN')
hybrid_mae, hybrid_mse, hybrid_r2, hybrid_files, hybrid_metrics_df = calculate_metrics(hybrid_dir, 'Hybrid')

metrics_df = pd.concat([gnn_metrics_df, hybrid_metrics_df], ignore_index=True)

print("Metrics for GNN Model (Normalized):")
print(f"  Average MAE: {gnn_mae}")
print(f"  Average MSE: {gnn_mse}")
print(f"  Average R²: {gnn_r2}")
print(f"  Number of Files: {gnn_files}\n")

print("Metrics for Hybrid Model (Normalized):")
print(f"  Average MAE: {hybrid_mae}")
print(f"  Average MSE: {hybrid_mse}")
print(f"  Average R²: {hybrid_r2}")
print(f"  Number of Files: {hybrid_files}")


sns.set(style="whitegrid")

avg_metrics = pd.DataFrame({
    'Model': ['GNN', 'Hybrid'],
    'Average MAE': [gnn_mae, hybrid_mae],
    'Average MSE': [gnn_mse, hybrid_mse],
    'Average R2': [gnn_r2, hybrid_r2]
})


avg_metrics_melted = avg_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Value', hue='Model', data=avg_metrics_melted)
plt.title('Average Normalized Metrics Comparison')
plt.ylabel('Value')
plt.show()

metrics_list = ['MAE', 'MSE', 'R2']

for metric in metrics_list:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y=metric, data=metrics_df)
    plt.title(f'Distribution of {metric} Across Files')
    plt.ylabel(metric)
    plt.show()
