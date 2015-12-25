require 'nn'
require 'rnn'
require 'data'


-----------------------------------------
----------------Load Data----------------
-----------------------------------------
print('Loading data ...')
-- local inputs, outputs, num_classes = load_data('data/Chiroptera2classes.csv')
-- local inputs, outputs, num_classes = load_data('data/Chiroptera.csv')
local inputs, outputs, num_classes, vocabSize = load_data('data/Heterokontophyta_Mammalia_out.csv')

print('Finished loading data')

-- inputs = {}
-- outputs = {}
-- class = 1
-- for i = 1, 1000 do
--     local input = {}
--     for i = 1,50 do input[i] = torch.rand(1) end
--     table.insert(inputs, input)
--     table.insert(outputs, class)
--     if i % 500 == 0 then class = class + 1 end
-- end
-- num_classes = 2

-----------------------------------------
----------------Build Model--------------
-----------------------------------------
local lr = 0.1
local num_epochs = 100
local inputSize = 1
local hiddenSize = 64
local outputSize = num_classes
-- local lstm = nn.LSTM(1, num_classes)

local lstm = nn.Sequential()
                :add(nn.Sequencer(nn.LookupTable(vocabSize, hiddenSize)))
                :add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize)))
                :add(nn.SelectTable(-1))
                :add(nn.Linear(hiddenSize, outputSize))
                :add(nn.LogSoftMax())
lstm:getParameters():uniform(-0.1,0.1)

-- local logsoftmax = nn.LogSoftMax()
-- local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
local criterion = nn.ClassNLLCriterion()

for e = 1, num_epochs do
    lstm:training()
    local err = 0

    local shuffle = torch.totable(torch.randperm(#inputs))

    for _, i in ipairs(shuffle) do
        -- forward propagation
        local input = inputs[i]
        local target = outputs[i]

        -- input:size(1) to give size as a lua number not torch Tensor
        -- for step = 1,input:size(1) do
            -- tensor fill because lstm:forward needs a torch Tensor not a lua number
            -- lstm:forward(torch.Tensor(1):fill(input[step]))
            -- gradOutputs[step] = torch.Tensor(num_classes):zero()
        -- end

        local output = lstm:forward(input)
        -- local output = logsoftmax:forward(lstm.output)
        err = err + criterion:forward(output, target)


        -- backpropagation
        -- logsoftmax:zeroGradParameters()
        local gradOutputs = criterion:backward(output, target)
        -- gradOutputs[input:size(1)] = logsoftmax:backward(input, gradOutputs[input:size(1)])
        
        -- for step = input:size(1),1,-1 do
            -- lstm:backward(torch.Tensor(1):fill(input[step]), gradOutputs[step])
        -- end
        lstm:backward(input, gradOutputs)

        -- todo make sure logSoftMax needs update parameters
        -- logsoftmax:updateParameters(lr)
        lstm:updateParameters(lr)
        -- lstm:forget()
        lstm:zeroGradParameters()
    end
    print ("Epoch " .. e .. " error: " .. err)
end


lstm:evaluate()
local right = 0
for i = 1,#inputs do
    local input = inputs[i]
    local target = outputs[i]

    -- for step = 1,input:size(1) do
        -- lstm:forward(torch.Tensor(1):fill(input[step]))
    -- end
    local output = lstm:forward(input)
    -- print(lstm.output)

    -- local output = logsoftmax:forward(lstm.output)
    -- r = ''
    -- for i = 1,#output do
        -- local _, result = output[i]:max(1)
    --     r = r .. result[1]
    -- end
    -- print(r)
    local _, result = output:max(1)

    print ('it was supposed to be ' .. target .. ' lstm got it ' .. result[1])
    if target == result[1] then right = right + 1 end
end

print ('accuracy is ' .. right .. '/' .. #inputs)
