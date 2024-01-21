import re
def codet5_output_to_patch(output, config):
    if config in ['CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT', 'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT']:
        return output.strip()
    elif config == 'CODET5_REFINE_CODEFORM_NOCOMMENT':
        try:
            stack = ['{']
            start_index = output.index('{')
            patch = output[: start_index + 1]
            for c in output[start_index + 1: ]:
                patch += c
                if c == '}':
                    top = stack.pop()
                    if top != '{':
                        return ''
                    if len(stack) == 0:
                        return patch.strip()
                elif c == '{':
                    stack.append(c)
            return ''
        except:
            print("Exception: " + output)
            return ''

def plbart_output_to_patch(output, config):
    output = re.sub('/\\*.*\\*/', '', output)
    if config in ['PLBART_SEQFORM_MASKFORM_NOCOMMENT', 'PLBART_SEQFORM_COMMENTFORM_NOCOMMENT']:
        stack = ['{']
        if '{' not in output:
            return ''
        start_index = output.index('{')
        patch = output[: start_index + 1]
        for c in output[start_index + 1: ]:
            patch += c
            if c == '}':
                top = stack.pop()
                if top != '{':
                    return ''
                if len(stack) == 0:
                    return patch.strip()
            elif c == '{':
                stack.append(c)
        return ''

def incoder_output_to_patch(output, config):
    if config in ['INCODER_COMPLETE_CODEFORM_NOCOMMENT', 'INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT']:
        """
        find the } that matches the first { in the output
        """
        output = output[len('<|endoftext|>'):]
        output = output.split('<|mask:0|>')
        if len(output) < 3:
            return ''
        output = output[0] + output[2]

        output = output.strip().split('\n')
        no_comment_output = [line for line in output if not line.strip().startswith('//')]
        output = '\n'.join(no_comment_output)
        stack = ['{']
        try:
            start_index = output.index('{')
            patch = output[: start_index + 1]
            for c in output[start_index + 1: ]:
                patch += c
                if c == '}':
                    top = stack.pop()
                    if top != '{':
                        return ''
                    if len(stack) == 0:
                        return patch.strip()
                elif c == '{':
                    stack.append(c)
            return ''
        except Exception as e:
            return ''
