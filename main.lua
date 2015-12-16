require 'nn'
require 'rnn'
require 'data'


-----------------------------------------
----------------Load Data----------------
-----------------------------------------
print('Loading data ...')
inputs, outputs, num_classes = load_data('data/Chiroptera2classes.csv')
print('Finished loading data')

-----------------------------------------
----------------Build Model--------------
-----------------------------------------
local lr = 0.1
local num_epochs = 30

local lstm = nn.LSTM(1, num_classes)
lstm:backwardOnline()

local logsoftmax = nn.LogSoftMax()
local criterion = nn.ClassNLLCriterion()

for i = 1, #inputs do
    -- forward propagation
    print('Doing forward propagation ...')
    local input = inputs[i]
    local target = outputs[i]
    local gradOutputs = {}

    -- input:size(1) to give size as a lua number not torch Tensor
    for step = 1,input:size(1) do
        -- tensor fill because lstm:forward needs a torch Tensor not a lua number
        lstm:forward(torch.Tensor(1):fill(input[step]))
        gradOutputs[step] = torch.Tensor(num_classes):zero()
    end
    
    local output = logsoftmax:forward(lstm.output)
    local err = criterion:forward(output, target)

    print ("error: " .. err)
 
    -- backpropagation
    print('Doing backpropagation ...')
    gradOutputs[input:size(1)] = criterion:backward(output, target)
    gradOutputs[input:size(1)] = logsoftmax:backward(input, gradOutputs[input:size(1)])
    
    for step = input:size(1),1,-1 do
        lstm:backward(torch.Tensor(1):fill(input[step]), gradOutputs[step])
    end

    lstm:updateParameters(lr)
    lstm:forget()
end


-- forward
-- gradOutputs1 = {}
-- gradOutputs2 = {}
-- for step=1,sent1.len do
--   local x = lt1:forward(torch.DoubleTensor(1):fill(sent1.words[step]))
--   rnn1:forward(x)
--   gradOutputs1[step] = torch.DoubleTensor(64):zero()
--   gradOutputs1[step] = torch.DoubleTensor(64):zero()
-- end
-- for step=1,sent2.len do
-- local x = lt2:forward(torch.DoubleTensor(1):fill(sent2.words[step]))
--   rnn2:forward(x)
--   gradOutputs2[step] = torch.DoubleTensor(64):zero()
--   gradOutputs2[step] = torch.DoubleTensor(64):zero()
-- end
-- root1 = rnn1.output
-- root2 = rnn2.output