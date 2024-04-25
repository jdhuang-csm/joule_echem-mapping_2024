def get_cycle(path):
    start_index = path.name.find('Cycle')
    return int(path.name[start_index:].split('_')[0].replace('Cycle', '').replace('.DTA', ''))


def get_test_id(path):
    start_index = path.name.find('ID=')
    return int(path.name[start_index:].split('_')[0].replace('ID=', '').replace('.DTA', ''))


def get_mode(path):
    # Test 1 files were named ChargeHybrid instead of just CHARGE
    return path.name.split('_')[1].replace('Hybrid', '').upper()
