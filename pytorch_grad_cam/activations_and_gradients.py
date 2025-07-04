class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    # 自动调用__call__()函数，获取正向传播的特征层A和反向传播的梯度A'
    def __init__(self, model, target_layers, reshape_transform):
        # 传入模型参数，申明特征层的存储空间（self.activations）
        # 和回传梯度的存储空间（self.gradients）
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        # 注意，上文指明目标网络层是是用列表存储的（target_layers = [model.down4]）
        # 源码设计的可以得到多层cam图
        # 这里注册了一个前向传播的钩子函数“register_forward_hook()”，其作用是在不改变网络结构的情况下获取某一层的输出，也就是获取正向传播的特征层
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))   #正向传播特征层
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))   #反向传播梯度

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    # 自动调用，会self.model(x)开始正向传播，注意此时并没有反向传播的操作
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    # handle要及时移除掉，不然会占用过多内存
    def release(self):
        for handle in self.handles:
            handle.remove()
