import sys

from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
from wavesAI.model.aussm import SSMSequenceClassifier
from wavesAI.utils.diagnostics import summarize_gradients, summarize_activations, print_weights

import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
import numpy as np


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            # print(fn, fn.requires_grad, dir(fn))
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                # print(u)
                node_name = 'Variable\n '+fn.name()+f" norm: {torch.norm(u.grad)}"
                dot.node(str(id(fn)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn] if gi is not None):
                    fillcolor = 'red'
                # print(fn_dict[fn])
                norm = np.mean([ torch.norm(gi).item() for gi in fn_dict[fn] if gi is not None ])
                # print(norm)
                # sys.exit()
                dot.node(str(id(fn)), f"{str(type(fn).__name__)} \n norm: {norm}", fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

if __name__ == '__main__':
    vocab_size = 12
    sequence_length = 4
    n_classes = 10
    batch_size = 2
    d_model = 16
    d_state = 16
    flags = "aou"
    mode = "last"
    num_layers = 6

    x = torch.randint(0, vocab_size, (batch_size, sequence_length), device=torch.device('cuda'))
    y = torch.randint(0, n_classes, size=(batch_size,), device=torch.device('cuda'))

    options = {
        "net": {
            "output_dim": n_classes,
            "n_layer": num_layers,
            "d_model": d_model,
            "d_state": d_state,
            "flags": flags,
            "mode": mode,
            "x_factor": 1/10000,
            "x_bias_factor": 1 / 10000
        }
    }

    net = SSMSequenceClassifier(vocab_size=vocab_size, **options['net'])
    net.to(torch.device('cuda'))

    with summarize_activations(net, infix=['layers'], verbose=True) as batch_diag:
        z = net.forward(x)
        z = z.squeeze()
        y = y.squeeze()

        loss = F.cross_entropy(z, y)

    get_dot = register_hooks(loss)
    loss.backward()
    dot = get_dot()
    dot.save('tmp.dot')

    print("=================== gradient summary ====================")
    # summarize gradients
    summarize_gradients(net)

    print("=================== weights summary ====================")
    print_weights(net)
