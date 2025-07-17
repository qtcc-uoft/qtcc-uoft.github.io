---
title: "How Legit is the Ending of HBO's Silicon Valley?"
date: 2023-07-10
---

 {% include figure.liquid loading="eager" path="assets/blog/silicon_valley/mv5bodqwodk5njcxof5bml5banbnxkftztgwmdmwmdgyntm40._v1_.jpg" class="img-fluid rounded z-depth-1" %} 

\[image credit - HBO or whoever\]

HBO's _Silicon Valley_ is an amusing television show about an eccentric computer scientist navigating the cut-throat politics of the show's fictionalized but authentic representation of the real-world tech Mecca where founders, investors, and industry personnel desperately fight for control of (and shares in) the technology of the future. If you haven't watched it, I would lightly recommend it. Spoilers for the show in the rest of this post.

This blog post is about exploring the legitimacy of the technology behind the ending of the show. Machine Learning Optimization theory, compression, and Reinforcement Learning will be involved so some background with machine learning will help but I will give a brief introduction to everything.

## Ending Recap

The show revolves around a single company, Pied Piper, and its various misadventures. In order to (I assume) get a better sampling of the diverse tech landscape Silicon Valley cultivates and to keep the show's topic material novel, Pied Piper frequently goes through dramatic and somewhat impractical changes in direction with regards to what area of technology they are focusing on. It starts out as a compression company (which a quick Google search easily shows [isn't really a thing](https://www.google.com/search?client=firefox-b-1-d&q=compression+company) - companies might have teams working on compression algorithms, and lots of companies use compression technology, but a company _just_ doing compression seems fake) but becomes a storage company, cloud service company, AI research startup, and probably some other stuff I forgot to mention.

This blog post is about exploring a particular plot device at the end of the show, and how legitimate the technology surrounding it is. As a recap, the last season is primarily focused on Pied Piper's attempts to make a new decentralized internet enabled by their strong compression technology, with a large-scale demonstration of their tech set to be showcased at a Burning Man style festival deep in the Nevada desert. However, once set up they realize that bottlenecks of their compression algorithm are preventing their system from succeeding (which seems like something you would've done some simulated tests or back of the napkin math on before, but I digress) which would doom the entire company and an upcoming telecommunications provider deal. Luckily, their founder Richard Hendrix has the idea to use one of their engineer's machine learning systems to optimize their compression, which suddenly and stunningly breaks the bottleneck holding their system back (in a couple minutes, mind you), saving the festival and for a short while, the company.

 {% include figure.liquid loading="eager" path="assets/blog/silicon_valley/russfest.jpg" class="img-fluid rounded z-depth-1" %} 

\[Above - the desert festival, called Russfest. Note the hologram.\]

In the next and final episode of the series, they discover this new AI system has broken all encryption software in its pursuit of strong compression, representing a catastrophic security risk. This leads them to intentionally botch the full launch of their new technology to prevent it from enabling terrorists to the world through hacking nuclear missile silos or whatever (I'm pretty sure nukes are airgapped, but we can probably agree all encryption ending would be troublesome).

During the festival, Richard says this:

Richard: "It occurred to me that our network is kind of like a series of paths, much like the trails ants use in order to forage for food. But then, I thought that we don't need better paths. We need better ants. You see? **Fuck gradient descent.**"

Dinesh: "What?"

Richard: "I don't need to use Gilfoyle's AI in order to improve middle‐out \[their compression algorithm\]. I need to use middle‐out to create a new AI. A symbolic AI. And now, it is **teaching itself how to optimize**."

Like most of the technology on the show there is a lot of creative liberty taken with the proper terminology and ideas they use, because explaining stochastic optimization theory or the proof of universal neural network approximation on a show made to make people laugh would be a colossal waste of everybody's time. But we can learn a couple important things from this clip:

- The big insight Richard had was to use the AI itself, and not Gradient Descent, to optimize their model.

- Working compression into their AI system, somehow, let it "teach itself to optimize"

- The AI goes from "something"? to a ""symbolic AI""

Understanding the ending of the show will require is to think a little bit about each of these, so let's talk about each of those in turn!

## Idea 1: ML as an Optimization Method

This is probably the most straightforward thing to analyze. Top level: you can use a kind of machine learning called **reinforcement learning** to teach a machine learning model how to learn better.

\*\*\* If you are familiar with supervised learning and reinforcement learning, you can skip to the header "background over"\*\*\*

This first requires a brief (hopefully recap) of how modern machine learning models (usually deep neural networks, which we know from previous dialogue that Gilfoyle's AI system on the show is) learn from data. Most machine learning is what is called **supervised learning**, which means learning to predict a particular **label** from a **sample** of data. Some examples:

- You have pictures of animals, and want your ML model to label them as dogs or cats

- You have information from people's health records, and want to predict their net income

Usually, the data that you try to make a prediction from is are called **features** and the thing you are trying to predict (which can be a member of a class like the dogs/cats or a value like net income) is called the **label**. How most ML works is that you have a lot of labeled data; that is, feature data with an associated label, which you try to "learn" the relationship from feature->label from. Then, you apply it to new data you do not have the features for.

An important part of this learning is that the **distribution** (read: the relationship between the features and labels) is **stationary** (read: that it doesn't change). This should intuitively make sense: if you were trying to learn the difference between cats and dogs, if the definition of each suddenly changed halfway through your training process and now Australian Shepherds were considered cats, that would make learning what images are cats and dogs quite difficult.

 {% include figure.liquid loading="eager" path="assets/blog.silicon_valley/kobi.jpg" class="img-fluid rounded z-depth-1" %} 

(above - an Australian Shepard, a kind of dog (or maybe a cat?))

The process that modern machine learning models learn from existing data is called **gradient descent**. Essentially, gradient descent is a mathematical algorithm for finding the **lowest value of a function** (and what inputs to that function give it) if your function is **differentiable** (you can take the derivative of your function with respect to the input variables). Machine learning identifies some **objective function** that represents how close a machine learning model is to being totally correct in its decision, and then uses gradient descent to find how the model should be configured to give the lowest total loss. \[Of course, this is a dramatic oversimplification - if you want to learn more about the nuts and bolts, [see here](https://optmlclass.github.io/notes/optforml_notes.pdf)\].

{% include figure.liquid loading="eager" path="assets/blog_silicon_valley/5c5a0-1ddjcoepshlsu7tff7lmyuq.webp" class="img-fluid rounded z-depth-1" %} 

\[Above - an intuitive visualization of gradient descent, where you are trying to find the bottom of your loss "landscape" by moving in the direction the slope of the hill gives you, representative of the gradient of your loss function. Image credit: https://towardsdatascience.com/an-intuitive-explanation-of-gradient-descent-83adf68c9c33\]

It is worth briefly noting that most machine learning models nowadays don't strictly use the gradient descent algorithm, but improved offshoots of it - however, when people say "gradient descent" or "stochastic gradient descent" (aka SGD) they often are referring to the category of family or algorithms, including modern variants like RMSProp or Adam. I assume Richard was doing the same thing here.

Returning to the show, what did Richard mean when he proudly proclaimed "fuck gradient descent" and that his AI is "teaching itself how to optimize?" We roughly know how machine learning works, but what "features" and "labels" could we use to teach a machine learning model that teaches other machine learning models how to optimize better than SGD or the most commonly used optimization method at the time of the show's finale, Adam? Well, that require _another_ kind of machine learning called **reinforcement learning**.

Very very briefly, reinforcement learning (or RL) is concerned with learning in a dynamic changing environment. For instance, if you want to learn how to play a video game well it is not as simple as supervised learning. Maybe one way to do it would be to generate data consisting of game information (your health, location, nearby enemies, whatever) and a corresponding best action to take. Then you could just learn from that like described above! The problem here is that your knowledge of the game changes as you play it - maybe after taking a kind of action in a certain scenario you notice that it was less good than you expected. If you just performed supervised learning, the efficacy of your game-playing agent would only be as strong as the person playing the game in your training data. RL gives you the ability to learn in such environments where your knowledge of the environment changes over time.


However, you can do a lot more with RL than just playing video games (although most of its greatest successes have come from gameplaying - see beating the world champions at the [Atari](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) games, [Dota II](https://openai.com/research/openai-five-defeats-dota-2-world-champions), [Go](https://www.deepmind.com/research/highlighted-research/alphago), and [Starcraft](https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning)). If you can formulate your problem as a sort of "game", there is a chance RL can overcome it. Most common are robotics based problems and financial applications.

\*\*\* Background Over \*\*\*

This is where I suspect Richard's insight came from - that he could use Gilfoyle's AI system to learn how to optimize itself better through Reinforcement Learning, instead of using a gradient descent based algorithm. There are papers doing this, a few of which I'll highlight:

- Li and Malik 16 treats the choice of update step in a SGD-style algorithm as a "policy" in reinforcement learning (you can think of a policy as a learned function from a state the agent is in to an action). They treated the current parameters and last _n_ gradients encountered as their state (they set n manually to 25) and the action as the parameter update step. They train their autonomous agent on a small set of small traditional machine learning iterative solvers (logictic and linear regressions), and later on neural networks in the follow-up Li and Malik 17. Figure 1 shows that their method, called "predicted step descent", works favorably on small neural networks, outperforming existing popular methods like Adam and Momentum-SGD. However, one limitation to this approach is that it requires the action space be equal to the number of parameters in your neural network, which can be difficult to learn. The paper is forced to overcome this by constraining their optimization problem in various ways. The fact that they don't report results on larger neural nets is also worrying. Another problem is that, via the fact that their policy is parameterized by a neural network, you cannot easily recover a closed-form update rule from your learning.

{% include figure.liquid loading="eager" path="assets/blog/silicon_valley/image.png" class="img-fluid rounded z-depth-1" %} 

Figure 1 - Learned optimization on a small neural net learning from CIFAR-100, from Li and Malik 17. Curves that go towards 0 in the y-axis more quickly are better.

- One way around both the large action space and no recoverable update rule problems is to treat the RL problem as one of finding the best symbolic update rule instead of the output of such a rule. In this problem formulation, instead of learning the policy directly you are learning the closed-form version of the optimal policy. This is what Bello et al. 17 do. They learn by searching over the combination space of a pre-defined set of mathematical operators applied to known useful gradient information, like the running average, running average squared, etc. which existing algorithms use. While this general approach of essentially using RL as equation space ablation might seem odd, similar approaches for other problem spaces occasionally find great success, like using RL to discover new provably asymptotically superior matrix-multiplication equations \[see Fawzi et al 22\]. The authors recover a number of seemingly both robust and intuitive optimizers, particularly highlighting `esign(g)*sign(m)*g`, which makes a more aggressive magnitude update in the gradient direction if the sign of the gradient and its moving average agree, and a smaller 1/e update if they disagree. They put this optimizer head to head with the standard Adam optimizer on a modern machine translation task, and find that their RL-learned algorithm is superior.

- Lastly, I will not go into this work in-depth but Harrison et al 22 provides a more modern summary of these kinds of optimization approaches, including discussions of the typical problems with learned optimization (that they are brittle and generalize poorly when a network is trained on significantly different data than what the optimizer learned on).

So it seems like we have the first piece of this puzzle figured out! Richard set Gilfoyle's AI system to optimize the compression technology crucial for their decentralized internet to function. Obviously there is some Hollywood magic here: Richard having this idea at the festival, integrating it live, and it magically working in a few minutes is less than realistic compared to people spending years on this problem and only finding very marginal improvements compared to existing optimization regimes. As impressive as the works I've shown are, none created an optimizer or meta-optimizer in wide-spread use. Most of the field (as far as I know) still uses Adam, despite the fact it [provably fails to converge on certain kinds of problems](https://arxiv.org/pdf/1904.09237.pdf) and is as of the writing of this post, nearly a decade old! Why is less clear. Some people think [neural network design evolved to fit Adam and not the other way around](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/), and some people just think Adam coincidentally happens to generalize to problems well in a way its predecessor SGD-Momentum or newer algorithms do not.

## Idea 2: ML for Compression

As a reminder of the show's context needed to understand this next point, the premise of Silicon Valley is based on Richard developing a groundbreaking compression algorithm, which is both space and time efficient and allows for search of the latent compressed space. We know by Season 6 he has improved this substantially through the [infamously named](https://youtu.be/Ex1JuIN0eaA) "middle out" and is using Gilfoyle's AI to optimize this compression further. Exactly how is not mentioned, but we know two things: 1\] from a discussion engineers at rival company Hooli have, we know it is somehow adaptive and learns to compress certain kinds of data better as it receives it online and 2\] that it uses a neural network, as per Gilfoyle's many comments about how his AI "Son of Anton" is a "black box" because it is an artificial neural network.

This is relevant here because the big problem Richard was trying to solve was insufficient compression hurting his decentralized internet system. As he says, he needed to make "better ants" (ie: to compress his data further so that more information could be transmitted along his network).

 {% include figure.liquid loading="eager" path="assets/blog/silicon_valley/MV5BMTkzNDYwNjA1N15BMl5BanBnXkFtZTgwMTU4NDIxMzE@._V1_.jpg" class="img-fluid rounded z-depth-1" %} 

\[Above - a poster from a far superior alternative to _a bug's life_\]

\[I will note here that while I already know only the basics about optimization theory and RL, I am an even greater neophyte in the field of compression - take everything here with a grain of salt\]

Because middle-out is a fake algorithm, it is hard to imagine the specifics of how machine learning might fit into it, but we know a little bit about it from the show. First, we know that middle-out is **lossless** from a conversation between Hooli engineers in the first episode of the show (this is a technically an algorithm from before middle-out's inception, but it's reasonable to assume Richard worked with the same paradigm in mind later). But we also know that it is frequently used for video streaming, so it might have some lossy variant for applications where perfect reconstruction is not important. I will go in depth on the lossless component though, because 1\] it seems to me like the lossless component is more important for the scenario in this context (network data) and 2\] lossy compression using neural networks would probably require talking about auto-encoders, which are very cool and useful but hard to explain without getting deep into the math, which I don't want to do.

For lossless compression, one can use what are called **auto-regressive models** combined with **arithmetic encoding**. In simple terms, arithmetic encoding is a way to store data in a way (imagine a string of characters) so that more-frequently used characters are represented with less bytes than infrequently used characters. It does this by imagining the entire compressed set of data as a single number between 0 and 1. For each character, we represent the probability distribution of the **most likely next character** as partitioning the range according to that distribution. For instance, i[f we just take the occurrence of letters in English as our distribution](https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html), I would give the letters E and A ~10% of the range from \[0,1) (say, from \[0.0,0.1) for A, and \[0.1,0.2) for E) but much less for infrequent letters like Z or Q. To encode the letter A, it would only require as many bits of precision needed to indicate a value in the \[0.0,0.1) range, but for Q (which accounts for ~0.2% of the English language) we would need the bits to account for a value in the \[9.998,1.000) range, which you can visually tell requires much more information! To encode another character, we take the range we have selected from encoding the previous character and repeat the process, except with the distribution over the previous range we had selected to encode our previous character, so if our first letter was A and was in the \[0.0,0.1) range, our new A would be in the \[0.0,0.01) range, E in the \[0.01,0.02) range, et cetera. [The Wikipedia page has more examples and helpful descriptions](https://en.wikipedia.org/wiki/Arithmetic_coding).


This is already quite efficient given even a naive distribution, but those of you familiar with the Transformer architecture can probably see an obvious way to improve this with a more intelligent (machine learned) approach to the distribution of our characters. Introduced in Attention is All you Need (Vaswami et al 17) , the Transformer architecture is a machine learning model very good at a specific kind of language modeling task called **next-token prediction**. This is a kind of supervised learning that takes in a series of text and wants to predict what character, word, or "token" (think bits of language larger than characters but smaller than words - for instance, the Vaswami paper treated the '-ing' modifier like in 'cooking' or 'playing' as a token) from the previous text it is given. You can then apply this to a large body of text in a supervised learning context by treating a selected sequence of text and its following token as your training data.

For instance, if I wanted to apply next-token prediction to the sentence "Mary had a little lamb" I would have 4 pieces of training data:

- feature "Mary" -> label "had"

- feature "Mary had" -> label "a"

- feature "Mary had a" -> label "little"

- feature Mary had a little" -> label "lamb".

In practice, the amount of words you can consider when predicting your next word (or the "context") is limited by memory constraints and model architecture choices. If you were unaware, this is the kind of model that powers Chat-GPT: it was trained on this next-token prediction task for a lot of data, and then was trained more on other specific tasks afterwards (called fine-tuning).

The neat trick comes from realizing that Transformer models are really just trying to learn the output distribution of tokens given context (they output a probability distribution of the token vocabulary), which is the exact thing we wanted to know in our arithmetic encoding compression regime above! Can we combine the two? Yes, to great success. One such implementation comes from Bellard 19, which achieves state of the art size compression on a large text data-set seen [here](http://www.mattmahoney.net/dc/text.html#1085). This work trained the Transformer model over just a single pass of the data before the compression pass, but you could train your model over a large training set if you wanted it to generalize to new data well, or even include some sort of online update so that it captured the changing underlying distribution of the incoming data you wanted to compress (you would need to decompress and re-compress it with your new distribution model, however). It seems likely given the characterization of Pied Piper's cloud platform that this is exactly what they are doing: using a neural network, trained on user data that they receive online, to improve compression sizes.

Could this apply to Richard's algorithm? It's not unfeasible that it could, if it was an algorithm like arithmetic encoding that benefits from some understanding of the underlying probabilistic structure the data represents. I choose to believe that it is either using a probabilistic model like the one I described above, or some similar schema to learn something about the data's distribution that can be cleverly used by his encoding scheme.

## Idea 3: Cleanup

This section is called "cleanup" because unlike the two above, it will not dive into interesting technical details but just address other small details of the comparison I am making. If you are only interested in the core thesis, you can skip to the conclusion.

Earlier I wrote that Gilfoyle/Pied Piper's AI goes from "something" to a "symbolic AI". I will address these each in turn.

What Gilfoyle's AI system _is exactly_ is one example where the show's lack of technical care (or more favorably, concern for entertainment over exact anal precision) is clear. Son of Anton (the name of Gilfoyle's AI/Neural Net/whatever) seems to be able to perform any kind of machine learning or AI adjacent task needed for the scene/plot. For instance, we see Gilfoyle try to use Son of Anton (unsuccessfully) to help him debug code. This is a thing real natural language processing models can do (like Github Copilot), but its more likely he is using an RL based system given that he and Dinesh talk about the "reward function being under-specified" which is firmly reinforcement-learning territory. But he also apparently uses Son of Anton as a chat-bot to impersonate Dinesh, which seems like more of an NLP task.

In reality, the ability for single machine learning system to _learn multiple tasks_ and a_pply learning across domains_ is an exceptionally difficult open problem usually called multi-modal learning. This is one area, I would say, where Silicon Valley is firmly in the realm of future advances for now - we have individual models that can do many of the things they show on the show ([one ML based app they showed on the show they even made in real life!](https://www.engadget.com/2017-05-15-not-hotdog-app-hbo-silicon-valley.html)) but a single system with such strong multi-modal capacity is an active research area. As of the time of this article's writing, [DeepMind's next big project is combining NLP and RL](https://www.wired.com/story/google-deepmind-demis-hassabis-chatgpt/) which seems eerily similar to how Son of Anton is portrayed in the show!

Also, what exactly is a "symbolic AI?" After doing some reading, I still have no idea what actually differentiates it from "subsymbolic" AI. Maybe the RL system from earlier that learned symbolic gradient descent rules counts because it returns a symbolic expression. I am also going to chalk this up to buzzwords and say it's not that important to the overall thesis.

## Conclusion

I have shown above that many of the technical components of the world-ending AI developed by accident in the end of Silicon Valley are actually surprisingly plausible. Even if the performance of the techniques they showed far exceed real-life in pursuit of a compelling story, they did not totally make them up, or at least put enough real words in the script to let one reconstruct the algorithm they are talking about.

However, this is where my thesis that the ending is somewhat plausible falls apart, due to one small unfortunate fact: I have no idea how learning superior compression would help one break general encryption. Now, the general idea that AI can learn something it wasn't "intended" to, potentially in a way that is harmful or dangerous, is not only reasonable but somewhat common (see Irpan 18 for examples in the RL context). But encryption and compression, while connected, are not so intertwined that if you had some system that could completely solve one it would break the other wide open as well. Maybe an encryption expert disagrees with me: if so, I would love to hear your opinion on this. But I cannot see a way for this to be plausible. It seems more likely to me that advances in quantum computing or pure math have some small risk of breaking encryption.

So in summary: how legit is the ending of Silicon Valley? The final result is not so legit, but a lot of the technology they describe along the way is surprisingly grounded in real machine learning. I give them maybe a B-/B overall. I hope you found this interesting, and thank you for reading!

Works Referenced

Li, Ke, and Jitendra Malik. "Learning to optimize." _arXiv preprint arXiv:1606.01885_ (2016).

Li, Ke, and Jitendra Malik. "Learning to optimize neural nets." _arXiv preprint arXiv:1703.00441_ (2017).

Bello, Irwan, et al. "Neural optimizer search with reinforcement learning." _International Conference on Machine Learning_. PMLR, 2017.

Fawzi, Alhussein, et al. "Discovering faster matrix multiplication algorithms with reinforcement learning." _Nature_ 610.7930 (2022): 47-53.

Harrison, James, Luke Metz, and Jascha Sohl-Dickstein. "A closer look at learned optimization: Stability, robustness, and inductive biases." _Advances in Neural Information Processing Systems_ 35 (2022): 3758-3773.

Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).

Bellard, Fabrice. "Lossless data compression with neural networks." _URL: https://bellard. org/nncp/nncp. pdf_ (2019).

Irpan, Alex. "Deep Reinforcement Learning Doesn't Work Yet". https://www.alexirpan.com/2018/02/14/rl-hard.html, 2018.

Other Interesting Works

Yang, Yibo, Stephan Mandt, and Lucas Theis. "An introduction to neural data compression." _Foundations and Trends® in Computer Graphics and Vision_ 15.2 (2023): 113-200. \[I got most of the information from the compression section from here! Well worth a read\]

The below Reddit post, which is in a very similar vein to this one.

https://www.reddit.com/r/SiliconValleyHBO/comments/e57hca/pipernet\_son\_of\_anton\_isnt\_quite\_nonsense/
