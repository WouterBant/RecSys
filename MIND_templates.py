
all_tasks = {}

# =====================================================
# Task Subgroup 1 -- sequential
# ====================================================
task_subgroup_1 = {}

template = {}

template['source1'] = "A user recently read articles {}, will the user read article {} ?"
template['target1'] = "{}"
template['task'] = "sequential"
template['source_argc1'] = 2
template['source_argv1'] = ['history', 'item']
template['target_argc1'] = 1
template['target_argv1'] = ['yes/no']
template['id'] = "1-1"

task_subgroup_1["1-1"] = template

all_tasks['sequential'] = task_subgroup_1



'''
"After reading articles {}, the user is interested in exploring more diverse topics. Will the user read article {} ?"




template = {}

template['source1'] = "A user recently read articles {}, will the user read article {} ?"
template['target1'] = "{}"
template['task'] = "sequential"
template['source_argc1'] = 2
template['source_argv1'] = ['history', 'item']
template['target_argc1'] = 1
template['target_argv1'] = ['yes/no']
template['id'] = "1-1"

task_subgroup_1["1-1"] = template



template = {}

template['source'] = "A user is interested in articles from categories {}, will he read article {} ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['category', 'item']
template['target_argc'] = 1
template['target_argv'] = ['yes/no']
template['id'] = "1-2"

task_subgroup_1["1-2"] = template
'''