import pandas as pd


file = "avg_evaluation_all.tsv"
df = pd.read_table(file)

df.loc[df['id'].str.contains('35ac16b9-928a-43...'), 'id'] = '35ac16b9-928a-43cd-b9b1-afd7fe07fa1f'
df.loc[df['id'].str.contains('2cd735e7-9390-41...'), 'id'] = '2cd735e7-9390-415e-b4ca-76505ec9dc52'
df.loc[df['id'].str.contains('c2917c1b-e4a8-49...'), 'id'] = 'c2917c1b-e4a8-495d-97fb-2593e5b168f6'
df.loc[df['id'].str.contains('b842e6ab-90eb-41...'), 'id'] = 'b842e6ab-90eb-4115-bc17-a3dc49abc619'
df.loc[df['id'].str.contains('e1f2c597-43ea-40...'), 'id'] = 'e1f2c597-43ea-4095-a3c8-f200063e0b8a'

df_config = df.sort_values(by='config')
df_config.to_csv("perturbation_sort_config.csv")

df_sample= df.sort_values(by='target_txt')
df_sample.to_csv("perturbation_sort_samples.csv")
