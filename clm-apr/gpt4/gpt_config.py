CodeT5InputConfig = {
    "CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT": {
        "model_id": "codet5-small/base/large",
        "input": "entire buggy function, with buggy lines masked by <extra_id_0>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
    "CODET5_REFINE_CODEFORM_NOCOMMENT": {
        "model_id": "codet5-refine",
        "input": "entire buggy function",
        "patch": "code generated by the model, which will replace the entire buggy function"
    },
    "CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "codet5-small/base/large",
        "input": "entire buggy function, with comments telling the buggy lines and buggy lines masked by <extra_id_0>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
}
def parse_method(method):
    try:
        for k in ['public', 'protected', 'private', 'default']:
            end_scope_index = method.find(k)
            if end_scope_index != -1:
                end_scope_index += len(k) + 1
                break
        index_final = method.find('static')
        if index_final != -1:
            end_scope_index = index_final + len('static') + 1
        index_final = method[:end_scope_index+1+len('final')].find('final')
        if index_final != -1:
            end_scope_index = index_final + len('final') + 1
        if end_scope_index == -1:
            end_scope_index = 0

        scope = method[:end_scope_index]
        inp_without_scope = method[end_scope_index:]

        name = inp_without_scope[:inp_without_scope.index('(')].split()[-1]

        ret_type = method[end_scope_index:method.index(name)]

        return scope, ret_type, name
    except:
        print("Error processing: "+ method)
        return "", "", ""