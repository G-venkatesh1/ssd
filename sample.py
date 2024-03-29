import ast

def find_named_tensors(file_path):
    named_tensors = []

    with open(file_path, 'r') as file:
        code = file.read()

    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_name'):
                    value = node.value
                    if isinstance(value, ast.Attribute) and value.attr == 'name':
                        named_tensors.append(target.id)
    
    return named_tensors

test_dataset_file = 'test_dataset.py'
named_tensors = find_named_tensors(test_dataset_file)

print("Named Tensors:")
for tensor in named_tensors:
    print(tensor)
