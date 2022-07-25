from sklearn.linear_model import LogisticRegression
from IPython.display import HTML
from tqdm import tqdm_notebook
#import sys; sys.path.append('paper')
#from initialize import * # user settings: feel free to change

#from initialize import * 
from recourse import *


import pandas as pd
# import data


from examples.paper.initialize import *

# user settings
settings = {
    #
    # audit settings
    'data_name': 'credit',
    'method_name': 'logreg',
    'normalize_data': True,
    'force_rational_actions': False,
    #
    # script flags
    'audit_recourse': True,
    'plot_audits': True,
    'print_flag': True,
    'save_flag': True,
    'randomseed': 2338,
    #
    # placeholders
    'method_suffixes': [''],
    'audit_suffixes': [''],
    }



	
# file names
output_dir = results_dir / settings['data_name']
output_dir.mkdir(exist_ok = True)

if settings['normalize_data']:
    settings['method_suffixes'].append('normalized')

if settings['force_rational_actions']:
    settings['audit_suffixes'].append('rational')

# set file header
settings['dataset_file'] = '%s/%s_processed.csv' % (data_dir, settings['data_name'])
settings['file_header'] = '%s/%s_%s%s' % (output_dir, settings['data_name'], settings['method_name'], '_'.join(settings['method_suffixes']))
settings['audit_file_header'] = '%s%s' % (settings['file_header'], '_'.join(settings['audit_suffixes']))
settings['model_file'] = '%s_models.pkl' % settings['file_header']
settings['audit_file'] = '%s_audit_results.pkl' % settings['audit_file_header']
pp.pprint(settings)



# data set
data_df = pd.read_csv(settings['dataset_file'])
data = {
    'outcome_name': data_df.columns[0],
    'variable_names': data_df.columns[1:].tolist(),
    'X': data_df.iloc[:, 1:],
    'y': data_df.iloc[:, 0]
    }

scaler = None
data['X_train'] = data['X']
data['scaler'] = None
if settings['normalize_data']:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
    data['X_scaled'] = pd.DataFrame(scaler.fit_transform(data['X'].to_numpy(dtype = float), data['y'].values), columns = data['X'].columns)
    data['X_train'] = data['X_scaled']
    data['scaler'] = scaler


	
from recourse.action_set import *

# action set
default_bounds = (1.0, 99.0, 'percentile')
custom_bounds = None
immutable_variables = []
#if settings['data_name'] == 'credit':

immutable_names = ['Female', 'Single', 'Married']
immutable_names += list(filter(lambda x: 'Age' in x or 'Overdue' in x, data['variable_names']))
default_bounds = (0.1, 99.9, 'percentile')
custom_bounds = {'Female': (0, 100, 'p'),  'Married': (0, 100, 'p')}
data['immutable_variable_names'] = [n for n in immutable_names if n in data['variable_names']]



## set the default and custom bounds we wish to search over.
default_bounds = (0.1, 99.9, 'percentile')
custom_bounds = {'Female': (0, 100, 'p'),  'Married': (0, 100, 'p')}

## 
action_set = ActionSet(X = data['X'], custom_bounds = custom_bounds, default_bounds = default_bounds)

action_set[data['immutable_variable_names']].actionable = False


for i, n in enumerate(action_set.name[6:14]):
    action_set[n].actionable = True 


#action_set = ActionSet(X = data['X'], custom_bounds = custom_bounds, default_bounds = default_bounds)
#action_set[data['immutable_variable_names']].mutable = False

action_set['EducationLevel'].step_direction = 1
payment_fields = list(filter(lambda x: 'Amount' in x, data['variable_names']))
action_set[payment_fields].step_type = 'absolute'
action_set[payment_fields].step_size = 50

for p in payment_fields:
    action_set[p].update_grid()



# model
model_stats = pickle.load(open(settings['model_file'], 'rb'))
all_models = model_stats.pop('all_models')

### Create Flipset
clf = all_models['C_0.02__max_iter_1000__penalty_l1__solver_saga__tol_1e-08']
yhat = clf.predict(X = data['X_train'])

coefficients, intercept = undo_coefficient_scaling(clf, scaler = data['scaler'])


y_desired = 1 
coefs = coefficients

flips = np.sign(coefs) if y_desired > 0 else -np.sign(coefs)

action_set.flip_direction = flips


#############################################################


action_set.set_alignment(coefficients)
predicted_neg = np.flatnonzero(yhat < 1)
U = data['X'].iloc[predicted_neg].values
k = 4

fb = Flipset(x = U[k], action_set = action_set, coefficients = coefficients, intercept = intercept)
fb.populate(enumeration_type = 'distinct_subsets', total_items = 14)
print(fb)




#################################################################################



model_stats_df = refomat_gridsearch_df(
    grid_search_df,
    settings=settings,
    n_coefficients = data['X'].shape[1],
    invert_C=settings['method_name'] == 'logreg'
)


audit_results = {}
for key, clf in model_dict.items():
    if settings['method_name'] == 'logreg':
        model_name = 1. / float(key.split('_')[1])
    else:
        model_name = float(key.split('_')[1])
        
    # unscale coefficients
    if scaler is not None:
        coefficients, intercept = undo_coefficient_scaling(coefficients = np.array(clf.coef_).flatten(), intercept = clf.intercept_[0], scaler = scaler)
    else:
        coefficients, intercept = np.array(clf.coef_).flatten(), clf.intercept_[0]

    ## run audit
    print("Auditing for model %s..." % key)
    auditor = RecourseAuditor(
        action_set,
        coefficients = coefficients,
        intercept = intercept
    )
    audit_results[model_name] = auditor.audit(X = data['X'])


## cache
if settings['save_flag']:
    pickle.dump(audit_results, file = open(settings['audit_file'], 'wb'), protocol=2)


###########################################################
# PLOTS

save_figs = False

## plotting performance vs. l1 penalty
train_error = model_stats_df.groupby('param 0: C')['train_score'].aggregate(['mean', 'var'])
test_error = model_stats_df.groupby('param 0: C')['test_score'].aggregate(['mean', 'var'])

ax = test_error.pipe(lambda df:
    (-df['mean']).plot(label='test error', color='black', figsize=(6, 6))
)
ax.errorbar(test_error.index, -test_error['mean'], yerr=test_error['var'], fmt='o', color='black')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# formatting
y_ticks = list(plt.yticks()[0])
y_tick_labels = list(map(lambda x: '%.02f' % (x * 100) + '%', y_ticks))
plt.semilogx(basex=10)
plt.yticks(y_ticks[::2], y_tick_labels[::2])
#
ax.set_ylabel('10-CV Mean Test Error')
ax.set_xlabel(xlabel)
if save_figs:
    plt.savefig('%s_error_path.pdf' % settings['file_header'], bbox_inches='tight')
    plt.close()



#########################################################

# plotting # of coefficients vs. l1 penalty...
# get binary counts and sums
coef_df = get_coefficient_df(all_models, variable_names = data['X'].columns.tolist())
nnz_coef_df = (coef_df
    .apply(lambda x: ~np.isclose(x, 0, atol=5e-3))
)
# cache how many features are nonzero for each classifier
non_zero_sum = nnz_coef_df.sum().rename(lambda x: 1./float(x.split('_')[1]))
non_zero_sum_actionable = (nnz_coef_df
                           .pipe(lambda df: df.loc[~df.index.isin(immutable_names)])
                           .sum()
                           ).rename(lambda x: 1./float(x.split('_')[1]))

# plot # of non zero coefficients
ax = non_zero_sum.plot(marker='o', label = 'All Variables', figsize=(6, 6))
ax = non_zero_sum_actionable.plot(ax= ax, marker='o', label='Actionable Variables')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.semilogx()
ax.set_yticks(list(map(int, ax.get_yticks())))
ax.set_facecolor('white')
ax.set_xlabel(xlabel)
ax.set_ylim((0, max(non_zero_sum.max(), non_zero_sum_actionable.max())*1.2))
ax.set_ylabel('# of Non-Zero Coefficients')
ax.legend(frameon = True)
if save_figs:
    f.savefig('%s_regularization_path.pdf' % settings['file_header'])
    plt.close()

#######################################################

if settings['method_name'] == 'logreg':
    xlabel = '$\ell_1$-penalty (log scale)'
else:
    xlabel = '$C$-penalty (log scale)'
# percent of points without recourse
feasibility_df = {}
obj_val = {}

for model_name in sorted(audit_results):
    recourse_df = pd.DataFrame(audit_results[model_name])
    recourse_cost = recourse_df.loc[lambda df: df.feasible.notnull()].loc[:, 'cost']
    feasibility_df[model_name] = recourse_df['feasible'].mean()
    obj_val[model_name] = recourse_cost.mean()

# feasibility plot
f, ax = create_figure(fig_size = (6, 6))
t_found = pd.Series(feasibility_df)
t_found.plot(ax = ax, color = 'black', marker='o')
plt.semilogx()
ax.set_xlabel(xlabel)
ax.set_ylabel('% of Individuals with Recourse')
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals = 0))
ax = fix_font_sizes(ax)
if save_figs:
    f.savefig('%s_recourse_feasibility.pdf' % settings['audit_file_header'], bbox_inches = 'tight')



##########################################################################

cost_df = {k: pd.DataFrame(v) for k, v in audit_results.items()}
cost_df = pd.concat([cost_df[k]['cost'].to_frame('%f' % k) for k in sorted(cost_df.keys())], axis=1).replace([-np.inf, np.inf], np.nan)

# plot cost distribution
f, ax = create_figure(fig_size = (6, 6))
sns.violinplot(data = cost_df, ax = ax, linewidth = 0.5, cut = 0, inner = 'quartile', color = "gold", scale = 'width')
ax.set_xlabel(xlabel)
ax.set_ylabel('Cost of Recourse')
ax.set_ylim(bottom = 0, top = 1)
xtick_labels = []
for xt in ax.get_xticklabels():
    v = np.log10(float(xt.get_text()))
    label = '$10^{%.0f}$' % v if v == np.round(v, 0) else ' '
    xtick_labels.append(label)
ax.set_xticklabels(xtick_labels)

for l in ax.lines:
    l.set_linewidth(3.0)
    l.set_linestyle('-')
    l.set_solid_capstyle('butt')

ax = fix_font_sizes(ax)
if save_figs:
    f.savefig('%s_recourse_cost_distribution.pdf' % settings['audit_file_header'], bbox_inches = 'tight')
    plt.close()



##########################################################################


