import batchGenerator

bg = batchGenerator.batchGenerator('features/input', 'features/output')
print bg.getBatch(0, 32)
