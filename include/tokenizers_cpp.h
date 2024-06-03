/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_cpp.h
 * \brief A C++ binding to common set of tokenizers
 */
#ifndef TOKENIZERS_CPP_H_
#define TOKENIZERS_CPP_H_

#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

/*!
 * \brief a universal tokenizer that loads
 *  either HF's tokenizer or sentence piece,
 *  depending on the constructor
 */
class Tokenizer {
 public:
  /*! \brief virtual destructor */
  virtual ~Tokenizer() {}

  /*!
   * \brief Encode text into ids.
   * \param text The input text.
   * \param add_special_tokens Whether or not to add special tokens when encoding the sequences.
   * \returns The encoded token ids.
   */
  virtual std::vector<int32_t> Encode(const std::string& text, bool add_special_tokens) = 0;

  /*!
   * \brief Encode text into ids.
   * \param text The input text.
   * \returns The encoded token ids.
   */
  virtual std::vector<int32_t> Encode(const std::string& text) { return Encode(text, false); }

  /*!
   * \brief Encode a batch of texts into ids.
   * \param texts The input texts.
   * \param add_special_tokens Whether or not to add special tokens when encoding the sequences.
   * \returns The encoded token ids.
   */
  virtual std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts,
                                                        bool add_special_tokens) {
    // Fall back when the derived class does not implement this function.
    std::vector<std::vector<int32_t>> ret;
    ret.reserve(texts.size());
    for (const auto& text : texts) {
      ret.push_back(Encode(text, add_special_tokens));
    }
    return ret;
  }

  /*!
   * \brief Encode a batch of texts into ids.
   * \param texts The input texts.
   * \returns The encoded token ids.
   */
  virtual std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts) {
    return EncodeBatch(texts, false);
  }

  /*!
   * \brief Decode token ids into text.
   * \param text The token ids.
   * \param skip_special_tokens Whether or not to remove special tokens in the decoding.
   * \returns The decoded text.
   */
  virtual std::string Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) = 0;

  /*!
   * \brief Decode token ids into text.
   * \param text The token ids.
   * \returns The decoded text.
   */
  virtual std::string Decode(const std::vector<int32_t>& ids) { return Decode(ids, false); }

  /*!
   * \brief Decode a batch of token ids into text.
   * \param ids The token ids.
   * \param skip_special_tokens Whether or not to remove special tokens in the decoding.
   * \returns The decoded text.
   */
  virtual std::vector<std::string> DecodeBatch(const std::vector<std::vector<int32_t>>& ids,
                                               bool skip_special_tokens) {
    // Fall back when the derived class does not implement this function.
    std::vector<std::string> ret;
    ret.reserve(ids.size());
    for (const auto& id : ids) {
      ret.push_back(Decode(id, skip_special_tokens));
    }
    return ret;
  }

  /*!
   * \brief Decode a batch of token ids into text.
   * \param ids The token ids.
   * \returns The decoded text.
   */
  virtual std::vector<std::string> DecodeBatch(const std::vector<std::vector<int32_t>>& ids) {
    return DecodeBatch(ids, false);
  }

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered.
   */
  virtual size_t GetVocabSize() = 0;

  /*!
   * \brief Convert the given id to its corresponding token if it exists. If not, return an
   * empty string.
   */
  virtual std::string IdToToken(int32_t token_id) = 0;

  /*!
   * \brief Convert the given token to its corresponding id if it exists. If not, return -1.
   */
  virtual int32_t TokenToId(const std::string& token) = 0;

  //---------------------------------------------------
  // Factory functions from byte-blobs
  // These factory function takes in in-memory blobs
  // so the library can be independent from filesystem
  //---------------------------------------------------
  /*!
   * \brief Create HF tokenizer from a single in-memory json blob.
   *
   * \param json_blob The json blob.
   * \return The created tokenzier.
   */
  static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string& json_blob);
  /*!
   * \brief Create BPE tokenizer
   *
   * \param vocab_blob The blob that contains vocabs.
   * \param merges_blob The blob that contains the merges.
   * \param added_tokens The added tokens.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobByteLevelBPE(const std::string& vocab_blob,
                                                         const std::string& merges_blob,
                                                         const std::string& added_tokens = "");
  /*!
   * \brief Create SentencePiece.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::string& model_blob);
  /*!
   * \brief Create RWKVWorldTokenizer.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobRWKVWorld(const std::string& model_blob);
};

}  // namespace tokenizers
#endif  // TOKENIZERS_CPP_H_
