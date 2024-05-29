
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



