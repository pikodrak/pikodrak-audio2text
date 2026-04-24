#include <jni.h>
#include "whisper.h"
#include <string>
#include <android/log.h>

#define TAG "whisper_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_cz_vytvarenicher_whisper_WhisperLib_initFromFile(
        JNIEnv *env, jobject /*thiz*/, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(modelPath, path);
    if (!ctx) {
        LOGE("Failed to init whisper context from %s", path);
        return 0L;
    }
    LOGI("Whisper context ready: %s", whisper_print_system_info());
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_cz_vytvarenicher_whisper_WhisperLib_transcribe(
        JNIEnv *env, jobject /*thiz*/, jlong ctxPtr,
        jfloatArray samples, jstring language) {
    auto *ctx = reinterpret_cast<whisper_context *>(ctxPtr);
    if (!ctx) return env->NewStringUTF("");

    jfloat *data = env->GetFloatArrayElements(samples, nullptr);
    jsize   len  = env->GetArrayLength(samples);
    const char *lang = env->GetStringUTFChars(language, nullptr);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = false;
    params.n_threads        = 4;

    bool autoDetect = (strcmp(lang, "auto") == 0);
    params.language        = autoDetect ? nullptr : lang;
    params.detect_language = autoDetect;

    int rc = whisper_full(ctx, params, data, static_cast<int>(len));

    env->ReleaseFloatArrayElements(samples, data, JNI_ABORT);
    env->ReleaseStringUTFChars(language, lang);

    if (rc != 0) {
        LOGE("whisper_full returned %d", rc);
        return env->NewStringUTF("");
    }

    int n = whisper_full_n_segments(ctx);
    std::string result;
    result.reserve(256);
    for (int i = 0; i < n; i++) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (text) result += text;
    }
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_cz_vytvarenicher_whisper_WhisperLib_free(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong ctxPtr) {
    auto *ctx = reinterpret_cast<whisper_context *>(ctxPtr);
    if (ctx) whisper_free(ctx);
}

JNIEXPORT jstring JNICALL
Java_cz_vytvarenicher_whisper_WhisperLib_systemInfo(
        JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(whisper_print_system_info());
}

} // extern "C"
