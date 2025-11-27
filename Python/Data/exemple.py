# Dans leur Jupyter Notebook:
import recup_data

# Charger les deux datasets directement depuis le cloud:
df_users = recup_data.recup_users()
df_travel = recup_data.recup_travel()

# df_users est prêt pour la modélisation !
print(df_users.shape)
print(df_users.head(10))
print(df_travel.head(10))