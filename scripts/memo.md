# TODO:
replay buffer ノード実装．
海馬的な役割．

・学習
bufferは一定以上batchがたまるとsignalをpublishし，
それをsubscribeしたagentはaction publishingを休止して
batch learning を開始する．
ーsrvにしてminibatchたちを１度にまとめて渡す
  pros: GPUつかえる  
  cons: 同期取れないと巨大データを何度もpubすることになる;topicの設計思想に反する
ーmsgとしてminibatch１個ずつを投げまくる
  pros: pub-sub回数多い
  cons: 偏った負荷はない
ー学習完了・minibatch要求signalのmsgを追加して同期しつつ
  pros: 確実
  cons: 余計なtopicが増える; 生物学的でない？
ーそもそもagentとbufferを分離しない(master branch)
  pros: GPUつかえる; pub-subなしですむ
  cons: 生物学的でない？
        おなじnodeの別subとすればagentにアクセスできて並列性も保てる？別コアではないか．

※学習時の行動履歴はmemoryに保存しない？（夢を覚えているのは単なる副作用なのか？）

・想起
criticがpublishするQ値（の予測誤差(いつ予測？)）または
入力信号の予測誤差に基づいてsignalをpublish,
bufferはそれをsubscribeして，想起される記憶をpublish.
環境からの入力とmeanをとる？

# DEBUG
params_grad の次元確認済み
  pendulumの場合のshape:
  actor.output[None,1], actor.params[10], actor.params_grad[10], 
  critic.action_grads[1]
　
