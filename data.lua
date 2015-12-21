require 'csvigo'

function load_data(fname)
  window = 5
  data = csvigo.load({path = fname, verbose = false})
  inputs = data.X
  outputs = data.Y
  char_map = {}
  class_map = {}
  char_idx = 0
  class_idx = 0
  X = {}
  Y = {}

  for i = 1, #inputs do
    if i == 14 then break end -- truncate data
    seq = inputs[i]
    seq_mapped = torch.Tensor(#seq):zero()
    -- seq_mapped = torch.Tensor(19):zero()
    for j = 1, #seq, window do
      -- if j == 20 then break end -- truncate data
      if char_map[string.sub(seq,j,j+window)] == nil then
        char_idx = char_idx + 1
        char_map[string.sub(seq,j,j+window)] = char_idx
      end
      seq_mapped[j] = char_map[string.sub(seq,j,j+window)]
    end
    table.insert(X,seq_mapped)
    
    class = outputs[i]
    if class_map[class] == nil then
      class_idx = class_idx + 1 
      class_map[class] = class_idx
    end
    table.insert(Y, class_map[class]) 
  end
  return X, Y, class_idx
end


