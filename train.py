import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

df = pd.read_csv("hatla2ee_cars_august_2025.csv")  

print("عدد الصفوف الكلي:", len(df))

# CONVERT THE PRICE AND MILEAGE TO DOUBLE 

df['price'] = (
    df['price'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('EGP', '', regex=False)
    .str.strip()
)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['mileage'] = (
    df['mileage'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('Km', '', regex=False)
    .str.strip()
)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')


# delete the wrong prices
def is_repdigit_price(x):
    if pd.isnull(x):
        return False
    s = str(int(x))
    return len(s) >= 4 and len(set(s)) == 1

mask_fake_price = df['price'].apply(is_repdigit_price)
n_fake_price = mask_fake_price.sum()
df = df[~mask_fake_price]
print(f"تم حذف {n_fake_price} صف بسعر وهمي (أرقام مكررة بالكامل)  -> الصفوف المتبقية: {len(df)}")

# Some brands which has a name of 2 words had a problem 

brand_fix = {
    'Alfa': 'Alfa Romeo',
    'Great': 'Great Wall',
    'Land': 'Land Rover',
    'Ssang': 'Ssang Yong',
    'King': 'King Long',
}
mask = df['company'].isin(brand_fix.keys())

def fix_row(row):
    correct_company = brand_fix[row['company']]
    title = str(row['title']).strip()
    rest = title[len(correct_company):].strip()
    year = row['year']
    if pd.notnull(year):
        year_str = str(int(year))
        if rest.endswith(year_str):
            rest = rest[:-len(year_str)].strip()
    return pd.Series({'company': correct_company, 'model': rest})

fixed_cols = df.loc[mask].apply(fix_row, axis=1)
df.loc[mask, 'company'] = fixed_cols['company']
df.loc[mask, 'model'] = fixed_cols['model']

# some models values moved wrong to year 

year_str = df['year'].astype(str).str.replace('.0', '', regex=False)
mask_year = year_str == df['model'].astype(str)

def fix_year(row):
    title = str(row['title']).strip()
    last_token = title.split()[-1]
    return float(last_token)

df.loc[mask_year, 'year'] = df.loc[mask_year].apply(fix_year, axis=1)

# make all the non numerical words normalized


for col in ['company', 'model' , 'color' , 'transmission']:
    df[col] = df[col].str.replace(r'[^A-Za-z0-9]', '', regex=True).str.capitalize()

# delete the title , location , date_posted , detail_link columns 

df = df.drop(columns=['title', 'location', 'date_posted', 'detail_link'])
print("تم حذف: title, location, date_posted, detail_link")

# delete the duplicated rows
n_dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"تم حذف {n_dupes} صف مكرر  -> الصفوف المتبقية: {len(df)}")


# there were some models with "Other " value so we will delete them 
n_other = (df['model'] == 'Other').sum()
df = df[df['model'] != 'Other']
print(f"تم حذف {n_other} صف model=Other  -> الصفوف المتبقية: {len(df)}")


# delete the rows where price is null
n_price_null = df['price'].isnull().sum()
df = df[df['price'].notnull()]
print(f"تم حذف {n_price_null} صف price مفقود  -> الصفوف المتبقية: {len(df)}")


n_year_null = df['year'].isnull().sum()
df = df[df['year'].notnull()]
print(f"تم حذف {n_year_null} صف year مفقود  -> الصفوف المتبقية: {len(df)}")

median_mileage = df['mileage'].median()
n_mileage_null = df['mileage'].isnull().sum()
df['mileage'] = df['mileage'].fillna(median_mileage)
print(f"تم ملء {n_mileage_null} صف في mileage بالوسيط ({median_mileage} كم)")

# convert prices to USD
df['price'] = df['price'] / 50
print("تم تحويل price من EGP لـ USD (بالقسمة على 50)")

# delete the prices greater than 1 million
n_price_outliers = (df['price'] > 1_000_000).sum()
df = df[df['price'] <= 1_000_000]
print(f"تم حذف {n_price_outliers} صف price outlier (>$1,000,000)  -> الصفوف المتبقية: {len(df)}")

# outliers with IQR for mileage
q1 = df['mileage'].quantile(0.25)
q3 = df['mileage'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 3 * iqr
n_mileage_outliers = (df['mileage'] > upper_bound).sum()
df = df[df['mileage'] <= upper_bound]
print(f"تم حذف {n_mileage_outliers} صف mileage outlier  -> الصفوف المتبقية: {len(df)}")

#حذف outliers
n_low_price = (df['price'] < 200).sum()
df = df[df['price'] >= 200]
print(f"تم حذف {n_low_price} صف price منخفض جدًا (<$200)  -> الصفوف المتبقية: {len(df)}")

# removing noise (cars have prices more than 5 doubles of its similar)
peer_group = ['company', 'model', 'year']
peer_size = df.groupby(peer_group)['price'].transform('size')
peer_median = df.groupby(peer_group)['price'].transform('median')
price_ratio = df['price'] / peer_median

mask_price_error = (peer_size >= 3) & (price_ratio > 5)
n_price_error = mask_price_error.sum()
df = df[~mask_price_error]
print(f"تم حذف {n_price_error} صف سعره أكبر من 5 أضعاف متوسط أشباهه (نفس الشركة/الموديل/السنة)  -> الصفوف المتبقية: {len(df)}")


# extract transmission from features column

def derive_transmission(features):
    if pd.isnull(features):
        return 'manual'
    return 'automatic' if 'automatic' in str(features).lower() else 'manual'

df['transmission'] = df['features'].apply(derive_transmission)
print("توزيع transmission :")
print(df['transmission'].value_counts())
for col in ['transmission']:
    df[col] = df[col].str.replace(r'[^A-Za-z0-9]', '', regex=True).str.capitalize()


# delete the features column after use it
df = df.drop(columns=['features'])
print("تم حذف عمود: features")

# save the new data
df.to_csv("hatla2ee_cars_cleaned.csv", index=False)
print(f"العدد النهائي للصفوف: {len(df)}")
print("تم الحفظ في: hatla2ee_cars_cleaned.csv")

# the training
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

model_df = df.copy()

categorical_cols = ['company', 'model', 'color', 'transmission']
for col in categorical_cols:
    model_df[col] = model_df[col].astype('category')

X = model_df.drop(columns=['price'])
y = model_df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = xgb_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print("نتائج تدريب موديل XGBoost للتنبؤ بسعر السيارة")
print("=" * 60)
print(f"R² (معامل التحديد): {r2:.4f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE : ${mae:,.2f}")

importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nأهم العوامل المؤثرة على السعر:")
print(importance)

xgb_model.save_model("car_price_xgb_model.json")
print("\nتم حفظ الموديل في: car_price_xgb_model.json")