# coding: utf-8
import fire
import keras
import tensorflow as tf
from tf2onnx import tfonnx
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import EfficientNetB2


def spy_model(k, name):
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        if name == "MobileNet":
            model = MobileNet()
        elif name == "EfficientNetB2":
            model = EfficientNetB2()
        else:
            raise ValueError("Unknown model name %r." % name)

        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=session.graph_def,
            output_node_names=[model.output.op.name])

    return graph_def, model


def spy_convert(graph_def, model):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def=graph_def, name='')

        def spy_convert_in():
            return tfonnx.process_tf_graph(tf_graph=graph,
                                    input_names=[model.input.name],
                                    output_names=[model.output.name])
        
        spy_convert_in()


def convert(name):    
    for k in [keras, tf.keras]:
        graph_def, model = spy_model(k, name)
        spy_convert(graph_def, model)


def profile(profiler="pyinstrument", name="EfficientNetB2"):
    """
    Profiles the conversion of a model.
    
    :param profiler: one of spy, pyinstrument, cProfile
    :param name: model to profile, MobileNet, EfficientNetB2
    """
    print("profile(%r, %r)" % (profiler, name))
    if profiler == "spy":
        convert(name)
    elif profiler == "pyinstrument":
        from pyinstrument import Profiler

        profiler = Profiler(interval=0.0001)
        profiler.start()

        convert(name)

        profiler.stop()

        print(profiler.output_text(unicode=False, color=False))
    elif profiler == "cProfile":
        import cProfile, pstats, io
        from pstats import SortKey

        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        raise ValueError("Unknown profiler %r." % profiler)
        

if __name__ == '__main__':
    fire.Fire(profile)
