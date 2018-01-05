# plt.plot(losslist)
# plt.show()




#


# def glvq_costfun(self, x, batch_labels):
#     xp = -2 * torch.mm(x, self.prototypes.t())
#     xp += torch.sum(x * x, 1).expand(xp.t().size()).t()
#     xp += torch.sum(self.prototypes * self.prototypes, 1).expand(xp.size())
#     # xp = torch.norm()
#     e = xp.size()[0]
#     p = xp.size()[1]
#
#     xp2 = xp.clone()
#     xp2.data.scatter_(1, batch_labels.view(e, 1).data.long(), float("inf"))
#     _, arg_d1s = torch.min(xp, 1)
#     d1s = xp.gather(1, batch_labels.view(e, 1))
#
#     d2s, arg_d2s = torch.min(xp2, 1)
#     res = torch.sum((d1s - d2s) / (d1s + d2s))
#     return res, arg_d1s




#
# # @profile
# def train(model, loader, epochs=3):
#     model.train()
#     for e in range(epochs):
#
#         start = time.time()
#         if CUDA:
#             train_loss = torch.zeros(1).cuda()
#         j = 0
#         if CUDA:
#             model.cuda()
#         for batch in loader:
#             startb = time.time()
#             # torch.cuda.synchronize()
#             if j > 100000000:
#                 break
#             j += 1
#             text, img_feat, target, sizes = Variable(batch['text']), \
#                                             Variable(batch['img_feat']), \
#                                             Variable(batch['target']), \
#                                             Variable(batch['size'])
#             # torch.cuda.synchronize()
#             if CUDA:
#                 text, img_feat, target, sizes = text.cuda(), \
#                                                 img_feat.cuda(), \
#                                                 target.cuda(), \
#                                                 sizes.cuda()
#
#             optimizer.zero_grad()
#             print(text.size())
#             if text.size(1) > 25:
#                 continue
#             # show_memusage(device=device)
#
#             text_prediction = model(text)
#             # distances = F.pairwise_distance(text_prediction, img_feat)
#             # xp = -2 * torch.mm(text_prediction, img_feat.t())
#             # xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
#             # xp += torch.sum(img_feat * img_feat, 1).expand(xp.size())
#
#             total_idx = 0
#             # exp1 = img_feat[0].view(1, -1).expand(sizes.data[0], 2048)
#
#             # startl = time.time()
#             # name = Variable(torch.zeros(1))
#             #
#             loss = Variable(torch.zeros(1))
#             if CUDA:
#             #     name = name.cuda()
#                 loss = loss.cuda()
#             # use gather instead of creating the exp1 vector
#             # for i, size in enumerate(sizes.data):
#             #     name += (F.cosine_similarity(text_prediction[total_idx: (total_idx + size)],
#             #                                           img_feat[i].view(1, -1).expand(size, 2048))).sum()
#             #     loss += torch.sqrt(xp[total_idx:total_idx+size, i]).sum()
#
#                 # exp1 = torch.cat((exp1, img_feat[i].view(1, -1).expand(size, 2048)), 0)
#                 # total_idx += size
#             st = time.time()
#             idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes.data[x])])
#             print('timeee -- ', time.time() - st)
#             # loss2 = F.cosine_similarity(text_prediction, exp1[sizes.data[0]:]).sum()
#             if CUDA:
#                 idx = idx.cuda()
#
#
#
#             loss3 = F.pairwise_distance(text_prediction, img_feat[idx]).sum()
#             # loss3 = F.cosine_similarity(text_prediction, img_feat[torch.cuda.LongTensor(idx)]).sum()
#             # print('loop', loss.data[0], name.data[0], loss2.data[0], loss3.data[0], time.time() - startl)
#             # print('loop', loss[0], name[0], loss[0], loss3.data[0], time.time() - startl)
#             train_loss += loss3.data[0]  # loss.data[0]
#             loss3.backward()
#             optimizer.step()
#
#             # del loss3, text, img_feat, target, sizes, text_prediction, idx
#             print('batch time', time.time() - startb)
#             # if e < 5:
#             #     for param_group in optimizer.param_groups:
#             #         param_group['lr'] += 0.000004
#             #         print(param_group['lr'])
#             # param_group['lr'] *= lr_decay
#
#         # losslist.append(train_loss)
#         print(train_loss)
#         print(time.time() - start)



# the following is slower, but might require a bit less memory

# for i, size in enumerate(sizes.data):
#     scores = torch.zeros(10)
#     for k in range(10):
#         dist = F.cosine_similarity(text_prediction[total_idx: (total_idx + size)],
#                                    img_feat[i*10 + k].view(1, -1).expand(size, 2048)).sum()
#         scores[k] = torch.mean(dist).data[0]
#     test_loss += scores[target.data[i]]
#     top1 += 1 if target.data[i] in scores.topk(1, largest=False)[1] else 0
#     top3 += 1 if target.data[i] in scores.topk(3, largest=False)[1] else 0
#     top5 += 1 if target.data[i] in scores.topk(5, largest=False)[1] else 0
#     predictions.append(scores)
#     N += 1
#     total_idx += size

#
# losss = torch.nn.CosineSimilarity()
# scores = torch.zeros(10)
# for k in range(10):
#     dist = losss(text_prediction[total_idx: (total_idx + size)],
#                  img_feat[i * 10 + k].view(1, -1).expand(size, 2048)).mean()
#     scores[k] = torch.mean(dist).data[0]
# test_loss += scores[target.data[i]]
# if len(target.size()) > 1:
#     target = target.view(-1)
# top1 += 1 if target.data[i] in scores.topk(1, largest=False)[1] else 0
# top3 += 1 if target.data[i] in scores.topk(3, largest=False)[1] else 0
# top5 += 1 if target.data[i] in scores.topk(5, largest=False)[1] else 0
#
#




#
# device = 0
# show_memusage(device=device)
#
# torch.cuda.set_device(device)
# model = models.resnet101(pretrained=False)
# model.cuda()
# criterion = nn.CrossEntropyLoss()
#
# volatile = False
#
# show_memusage(device=device)
# for ii in range(3):
#     inputs = torch.randn(20, 3, 224, 224)
#     labels = torch.LongTensor(range(20))
#     inputs = inputs.cuda()
#     labels = labels.cuda()
#     inputs = Variable(inputs, volatile=volatile)
#     labels = Variable(labels, volatile=volatile)
#
#     print("before run model:")
#     show_memusage(device=device)
#     outputs = model(inputs)
#     print("after run model:")
#     show_memusage(device=device)
#     loss = criterion(outputs, labels)
#
#     if bool(int(sys.argv[1])):
#         print("before backward:")
#         show_memusage(device=device)
#         loss.backward()
#         print("after backward:")
#         show_memusage(device=device)
    #     del loss
    #     del outputs
    #     del inputs
    #     del labels
    #     show_memusage(device=device)
    # else:
    #     del loss
    #     del outputs
    #     # del inputs
    #     # del labels
    #     print "after delete:"
#     show_memusage(device=device)


# for i, size in enumerate(sizes):
#     if i == 0:
#         if config.cosine_similarity:
#             # loss = cosine_similarity(out[:size, i], img_prediction[i]).sum()
#             loss = - F.cosine_similarity(out[:size, i],
#                                          img_prediction[i].view(1, -1).expand(size, img_prediction.size(1))).sum()
#         else:
#             # aa = mean_squared_error(out[:size, i], img_prediction[i]).sum()
#             # aa = mean_squared_error(out[:size, i], img_prediction[i]).sum()
#             loss = torch.pow(
#                 F.pairwise_distance(out[:size, i], img_prediction[i].view(1, -1).expand(size, img_prediction.size(1))),
#                 2).sum()
#     else:
#         if config.cosine_similarity:
#             # loss += cosine_similarity(out[:size, i], img_prediction[i]).sum()
#             loss += - F.cosine_similarity(out[:size, i],
#                                           img_prediction[i].view(1, -1).expand(size, img_prediction.size(1))).sum()
#         else:
#             # aa = mean_squared_error(out[:size, i], img_prediction[i]).sum()
#             loss += torch.pow(
#                 F.pairwise_distance(out[:size, i], img_prediction[i].view(1, -1).expand(size, img_prediction.size(1))),
#                 2).sum()
