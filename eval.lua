require "image"


recon_image = image.load("2.png",1,'byte')
ground_truth = image.load("frame10i11.jpg",1,'byte')

recon_image = recon_image:double():mul(1./255.)
ground_truth = ground_truth:double():mul(1./255.)


residual = recon_image - ground_truth

print(residual[{ {}, {100},{}}])

image.save("resudual.png",residual:mul(255))
image.save("resudual.png",residual:mul(255))
