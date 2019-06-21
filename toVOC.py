import xml.dom.minidom

def createObject(name,box,dom,root):
    obj = dom.createElement("object")
    namel = dom.createElement("name")
    namel.appendChild(dom.createTextNode(name))
    obj.appendChild(namel)
    bnd = dom.createElement("bndbox")
    xmin = dom.createElement("xmin")
    xmin.appendChild(dom.createTextNode(str(box[0])))
    bnd.appendChild(xmin)
    ymin = dom.createElement("ymin")
    ymin.appendChild(dom.createTextNode(str(box[1])))
    bnd.appendChild(ymin)
    xmax = dom.createElement("xmax")
    xmax.appendChild(dom.createTextNode(str(box[2])))
    bnd.appendChild(xmax)
    ymax = dom.createElement("ymax")
    ymax.appendChild(dom.createTextNode(str(box[3])))
    bnd.appendChild(ymax)
    obj.appendChild(bnd)
    root.appendChild(obj)

def createAnnotation(index,names,boxes):
    dom = xml.dom.minidom.Document()
    root = dom.createElement("annocation")
    dom.appendChild(root)
    for i in range(len(names)):
        createObject(names[i],boxes[:,i],dom,root)

# DOMオブジェクトの生成
dom = xml.dom.minidom.Document()
root = dom.createElement("annotation")
dom.appendChild(root)
createObject("hoge",[1,2,3,4],dom,root)
createObject("fuga",[5,6,7,8],dom,root)
"""
# rootノードの生成と追加
root = dom.createElement('root')
dom.appendChild(root)

# サブノードの生成
subnode = dom.createElement('subnode')
subnode.appendChild(dom.createTextNode("日本語もOK"))
# サブノートにattributeとvalueを設定
subnode_attr = dom.createAttribute('key')
subnode_attr.value = 'value'
subnode.setAttributeNode(subnode_attr)
# itemノードにsubnodeノードを追加
root.appendChild(subnode)
"""
# domをxmlに変換して整形
print (dom.toprettyxml())
