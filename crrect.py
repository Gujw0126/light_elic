import torch
from compressai.compressai.zoo import load_state_dict
from teacher_models import ELICHyper
from compressai.compressai.models.utils import update_registered_buffers,remap_old_keys
from compressai.compressai.entropy_models import EntropyBottleneck, GaussianConditional

net_cls = ELICHyper()
state_dict = load_state_dict(torch.load("/mnt/data3/jingwengu/ELIC_light/hyper/lambda2/ELICHyper2.pth.tar", map_location="cuda"))
state_dict_net = net_cls.state_dict()

update_registered_buffers(
    net_cls.gaussian_conditional,
    "gaussian_conditional",
    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
    state_dict_net,
)


for name, module in net_cls.named_modules():
    if not any(x.startswith(name) for x in state_dict_net.keys()):
        continue

    if isinstance(module, EntropyBottleneck):
        update_registered_buffers(
            module,
            name,
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict_net,
        )
        state_dict_net = remap_old_keys(name, state_dict_net)

    if isinstance(module, GaussianConditional):
        update_registered_buffers(
            module,
            name,
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict_net,
        )


for key in state_dict_net.keys():
    state_dict_net[key] = state_dict[key]

torch.save(state_dict_net,"/mnt/data3/jingwengu/ELIC_light/hyper/lambda2/correct.pth.tar")


