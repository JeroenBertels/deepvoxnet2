from deepvoxnet2.components.model import Creator


def unique_transformer_names(creator):
    trans_ids = {}
    nb_transformers_present = {}
    for at in reversed(creator.active_transformers):
        at_name = type(at).__name__
        mod_names = ''
        if 'MircInput' in at_name:
            for mn in at.modality_ids:
                mod_names = mod_names + mn + ', '
            mod_names  = mod_names[:-2]
        if at_name not in nb_transformers_present.keys():
            nb_transformers_present[at_name] = 1
        else:
            nb_transformers_present[at_name] = nb_transformers_present[at_name] + 1
        final_name = at_name + '_{}'.format(nb_transformers_present[at_name]-1)
        if mod_names != '':
            final_name = final_name + ' (' + mod_names + ')'
        trans_ids[id(at)] = final_name
    return trans_ids


def print_path_information(processing_path):
    creator = Creator(processing_path)
    done_transformers = []
    print('_' * 135)
    print('{:<30} {:<5} {:<50} {:<5} {:<35} {:<5}'.format('Transformer', '#I', 'Input(s)', '#O', 'Output shape(s)', 'n'))
    print('=' * 135)
    trans_id_names = unique_transformer_names(creator)
    for ac in reversed(creator.active_connections):
        ac_transformer_id = id(ac.transformer)
        if ac_transformer_id in done_transformers:
            continue
        else:
            done_transformers.append(ac_transformer_id)
            trans_name = trans_id_names[ac_transformer_id]
            nb_outputs = len(ac.transformer.active_indices)
            active_inputs = ''
            nb_inputs = 0
            if len(ac.transformer.connections) != 0:
                for ai in ac.transformer.active_indices:
                    for li in ac.transformer.connections[ai]:
                        nb_inputs = nb_inputs + 1
                        if trans_id_names[id(li.transformer)] not in active_inputs:  # only add the active input if not yet present in the string
                            active_inputs = trans_id_names[id(li.transformer)] + ', ' + active_inputs
                active_inputs = active_inputs[:-2]
            else:
                active_inputs = ''
            output_shapes = []
            if not isinstance(ac.transformer.output_shape[0], tuple):
                output_shapes.append(ac.transformer.output_shape)
            else:
                for os in ac.transformer.output_shape:
                    output_shapes.append(os)
            n = ac.transformer.n if ac.transformer.n is not None else ''

            first_line = True
            for os in output_shapes:
                if first_line:
                    first_line = False
                    print('{:<30} {:<5} {:<50} {:<5} {:<35} {:<5}'.format(trans_name, nb_inputs, active_inputs, nb_outputs, str(os), n))
                else:
                    print('{:<30} {:<5} {:<50} {:<5} {:<35} {:<5}'.format('', '', '', '', str(os), ''))
            print('_' * 135)
    print('')