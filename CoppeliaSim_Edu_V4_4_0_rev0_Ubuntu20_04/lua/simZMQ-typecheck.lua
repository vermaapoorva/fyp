-- simZMQ lua type-checking wrapper
-- (this file is automatically generated: do not edit)
require 'checkargs'

local simZMQ=require('simZMQ')

function simZMQ.__addTypeCheck()
    local function wrapFunc(funcName,wrapperGenerator)
        _G['simZMQ'][funcName]=wrapperGenerator(_G['simZMQ'][funcName])
    end

end

sim.registerScriptFuncHook('sysCall_init','simZMQ.__addTypeCheck',true)

return simZMQ