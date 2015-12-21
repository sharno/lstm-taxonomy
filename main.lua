require 'nn'
require 'rnn'
require 'data'


-----------------------------------------
----------------Load Data----------------
-----------------------------------------
print('Loading data ...')
inputs, outputs, num_classes = load_data('data/Chiroptera2classes.csv')
-- inputs, outputs, num_classes = load_data('data/Chiroptera.csv')
print('Finished loading data')

-----------------------------------------
----------------Build Model--------------
-----------------------------------------
local lr = 0.03
local num_epochs = 100

local lstm = nn.LSTM(1, num_classes)
lstm:backwardOnline()

local logsoftmax = nn.LogSoftMax()
local criterion = nn.ClassNLLCriterion()
for e = 1, num_epochs do
    local err = 0

    local shuffle = torch.totable(torch.randperm(#inputs))

    for _, i in ipairs(shuffle) do
        -- forward propagation
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
        err = err + criterion:forward(output, target)


        -- backpropagation
        gradOutputs[input:size(1)] = criterion:backward(output, target)
        gradOutputs[input:size(1)] = logsoftmax:backward(input, gradOutputs[input:size(1)])
        
        for step = input:size(1),1,-1 do
            lstm:backward(torch.Tensor(1):fill(input[step]), gradOutputs[step])
        end

        -- todo make sure logSoftMax needs update parameters
        logsoftmax:updateParameters(lr)
        lstm:updateParameters(lr)
        lstm:forget()
    end
    print ("Epoch " .. e .. " error: " .. err)
end

for i = 1,#inputs do
    lstm:forget()

    local input = inputs[i]
    local target = outputs[i]

    for step = 1,input:size(1) do
        lstm:forward(torch.Tensor(1):fill(input[step]))
    end
    -- print(lstm.output)

    local output = logsoftmax:forward(lstm.output)
    -- print(output)
    local _, result = output:max(1)
    -- print(_)

    print ('it was supposed to be ' .. target .. ' lstm got it ' .. result[1])
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