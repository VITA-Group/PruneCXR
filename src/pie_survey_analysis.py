import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, kruskal

def normalize(x):
	return (x - x.min()) / (x.max() - x.min())

def standardize(x):
	return (x - x.mean()) / x.std()

def get_agree_response(x):
	try:
		response = x['Do you fully agree with the label?']
		return 1 if response == 'yes' else 0
	except:
		return np.nan

def get_quality_response(x):
	try:
		response = x['How would you rate the image quality?']
		return int(response)
	except:
		return np.nan

def get_difficulty_response(x):
	try:
		response = x['How difficult would you consider it to properly diagnose this image?']
		return int(response)
	except:
		return np.nan

def load_survey_data(path):
	x = pd.read_csv(path)
	x = x.groupby('filename', as_index=False).agg('first').reset_index(drop=True)  # remove duplicate rows	

	x['pie'] = ['PIE']*20 + ['Non-PIE']*20

	x['response_dict'] = x['file_attributes'].apply(lambda x: json.loads(x))

	x['agree_label'] = x['response_dict'].apply(get_agree_response)
	x['quality'] = x['response_dict'].apply(get_quality_response)
	x['difficulty'] = x['response_dict'].apply(get_difficulty_response)

	x['norm_quality'] = normalize(x['quality'])
	x['norm_difficulty'] = normalize(x['difficulty'])

	x['std_agree_label'] = standardize(x['agree_label'])
	x['std_quality'] = standardize(x['quality'])
	x['std_difficulty'] = standardize(x['difficulty'])

	return x

# Gather, clean, and preprocess survey data (collected in 7 separate forms/parts)
surveys = []
for i in range(1, 7):
	path = f'020523_PIE_CXR_v{i}_csv.csv'

	survey = load_survey_data(path)
	
	print('---', path, '---')
	print(survey.groupby('pie')[['agree_label', 'quality', 'difficulty']].agg(['mean', 'std']))
	print(survey.groupby('pie')[['std_agree_label', 'std_quality', 'std_difficulty']].agg(['mean', 'std']))

	survey['batch'] = i
	surveys.append(survey)
surveys = pd.concat(surveys, axis=0)

# Descriptive statistics of label agreement, image quality, and diagnosis difficulty ratings
for c in ['agree_label', 'quality', 'difficulty']:
	print(surveys.groupby('pie')[c].agg(['mean', 'std']))

# Significance tests for differences in rater responses for PIEs vs. non-PIEs
for col in ['agree_label', 'quality', 'difficulty']:
	pie_vals = surveys[surveys['pie'] == 'PIE'][col].values
	non_pie_vals = surveys[surveys['pie'] != 'PIE'][col].values

	t_test = ttest_ind(pie_vals, non_pie_vals, equal_var=False)
	mwu = mannwhitneyu(pie_vals, non_pie_vals)
	kw = kruskal(pie_vals, non_pie_vals)

	print('---', col, '---')
	print(f'\tP = {kw.pvalue} (Krusal-Wallis Test)')

# Plot mean +/- std radiologist response for (left) label agreement, (middle) image quality, and (right) diagnosis difficulty
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharey=True)

sns.pointplot(ax=ax[0], data=surveys, y='pie', x='agree_label', errorbar='sd', capsize=0.1, scale=1.5)
ax[0].set_ylabel('')
ax[0].set_yticklabels(['PIE', 'Non-PIE'], fontsize=13, rotation=90, va='center')
ax[0].set_xlabel('Radiologist Score\n(0=No, 1=Yes)', fontsize=12)
ax[0].set_title('Label Correctness', fontsize=14)

sns.pointplot(ax=ax[1], data=surveys, y='pie', x='quality', errorbar='sd', capsize=0.1, scale=1.5)
ax[1].set_ylabel('')
ax[1].set_title('Image Quality', fontsize=14)
ax[1].set_xlabel('Radiologist Score\n(1=Poor, ..., 5=Excellent)', fontsize=12)

sns.pointplot(ax=ax[2], data=surveys, y='pie', x='difficulty', errorbar='sd', capsize=0.1, scale=1.5)
ax[2].set_ylabel('')
ax[2].set_title('Diagnosis Difficulty', fontsize=14)
ax[2].set_xlabel('Radiologist Score\n(1=Very Easy, ..., 5=Very Difficult)', fontsize=12)

fig.tight_layout()
fig.savefig('cxr_pie_summary.pdf', bbox_inches='tight')