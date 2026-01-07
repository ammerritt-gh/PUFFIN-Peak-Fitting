import sys, json
sys.path.insert(0, r'c:/Users/AMM/Documents/Github/PUFFIN/PUFFIN')
from viewmodel.fitter_vm import FitterViewModel

vm = FitterViewModel()
vm.set_model('Custom Model')
vm.add_component_to_model('Gaussian')

specs = vm.get_parameters()
print('PARAMS:', list(specs.keys()))
# find the Area entry
k = next((kk for kk in specs.keys() if 'Area' in kk), None)
print('FOUND_AREA_KEY:', k)
if k is None:
    sys.exit(2)

# mark fixed via apply_parameters using the __fixed key
vm.apply_parameters({k + '__fixed': True})

ms = vm.state.model_spec
print('\nFLAT PARAM FIXED FLAGS:')
for kk, vv in ms.params.items():
    print(f"  {kk}: fixed={getattr(vv, 'fixed', None)} value={getattr(vv, 'value', None)}")

link = ms.get_link(k)
print('\nLINK:', link)
if link:
    comp, pname = link
    print('\nUNDERLYING COMPONENT PARAMS:')
    for kk, vv in comp.spec.params.items():
        print(f"  component {kk}: fixed={getattr(vv,'fixed', None)} value={getattr(vv,'value', None)}")
else:
    print('No link found')

# attempt to run a fit (non-blocking start) and report worker creation
vm.run_fit()
print('\nFIT_WORKER:', type(vm._fit_worker), 'status:', vm._fit_worker is not None)

print('\nDONE')
